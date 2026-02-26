import tensorflow as tf
from tensorflow.keras import Model, layers

PAD_ID = 0
START_ID = 2


class SeqVAE(Model):
    def __init__(
        self,
        vocab_size,
        emb_dim=128,
        enc_units=256,
        latent_dim=128,
        dec_units=256,
        max_len=200,
        dropout=0.2,
        gru_dropout=0.1,
        pooling_heads=4,
    ):
        super().__init__()
        self.max_len = max_len
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size
        self.pooling_heads = pooling_heads

        # ---- Shared token/position embeddings ----
        self.embedding = layers.Embedding(vocab_size, emb_dim, mask_zero=True)
        self.pos_embedding = layers.Embedding(max_len, emb_dim)
        self.embed_dropout = layers.Dropout(dropout)

        # ---- Encoder: local motif modeling + sequence context ----
        self.encoder_conv_3 = layers.Conv1D(
            filters=enc_units // 2,
            kernel_size=3,
            padding="same",
            activation="swish",
        )
        self.encoder_conv_7 = layers.Conv1D(
            filters=enc_units // 2,
            kernel_size=7,
            padding="same",
            activation="swish",
        )
        self.encoder_bi_gru = layers.Bidirectional(
            layers.GRU(
                enc_units // 2,
                return_sequences=True,
                dropout=gru_dropout,
                recurrent_dropout=gru_dropout,
            )
        )
        self.encoder_norm = layers.LayerNormalization()
        self.encoder_attn_score = layers.Dense(enc_units, activation="tanh")
        self.encoder_attn_logits = layers.Dense(pooling_heads)
        self.encoder_head_flatten = layers.Flatten()
        self.encoder_post_pool = layers.Dense(enc_units, activation="swish")

        self.z_mean = layers.Dense(latent_dim)
        self.z_log_var = layers.Dense(latent_dim)

        # ---- Decoder ----
        self.latent_to_context = layers.Dense(emb_dim, activation="tanh")
        self.latent_to_init_h = layers.Dense(dec_units, activation="tanh")
        self.decoder_gru = layers.GRU(
            dec_units,
            return_sequences=True,
            return_state=True,
            dropout=gru_dropout,
            recurrent_dropout=gru_dropout,
        )
        self.decoder_norm = layers.LayerNormalization()
        self.output_dense = layers.Dense(vocab_size)

    def _embed_tokens(self, token_ids, training=False):
        batch = tf.shape(token_ids)[0]
        seq_len = tf.shape(token_ids)[1]
        tok_emb = self.embedding(token_ids)
        pos_idx = tf.range(seq_len)[tf.newaxis, :]
        pos_emb = self.pos_embedding(pos_idx)
        pos_emb = tf.tile(pos_emb, [batch, 1, 1])
        x = tok_emb + pos_emb
        return self.embed_dropout(x, training=training)

    def encode(self, x, training=False):
        x_emb = self._embed_tokens(x, training=training)
        x_local = tf.concat(
            [self.encoder_conv_3(x_emb), self.encoder_conv_7(x_emb)], axis=-1
        )
        x_ctx = self.encoder_bi_gru(x_local, training=training)
        x_ctx = self.encoder_norm(x_ctx + x_local)

        mask = tf.cast(tf.not_equal(x, PAD_ID), tf.float32)
        attn_hidden = self.encoder_attn_score(x_ctx)
        attn_logits = self.encoder_attn_logits(attn_hidden)
        attn_mask = (1.0 - mask[:, :, tf.newaxis]) * -1e9
        attn_weights = tf.nn.softmax(attn_logits + attn_mask, axis=1)
        
        head_context = tf.einsum("bsh,bsd->bhd", attn_weights, x_ctx)
        h = self.encoder_head_flatten(head_context)
        h = self.encoder_post_pool(h)

        return self.z_mean(h), self.z_log_var(h)

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(tf.shape(mean))
        return mean + tf.exp(0.5 * logvar) * eps

    def _decode_teacher_forcing(self, z, target_ids, training=False):
        batch = tf.shape(z)[0]
        start = tf.fill([batch, 1], START_ID)
        shifted = tf.concat([start, target_ids[:, :-1]], axis=1)

        x = self._embed_tokens(shifted, training=training)
        z_context = self.latent_to_context(z)[:, tf.newaxis, :]
        x = x + z_context

        init_h = self.latent_to_init_h(z)
        h_seq, _ = self.decoder_gru(x, initial_state=init_h, training=training)
        h_seq = self.decoder_norm(h_seq)
        return self.output_dense(h_seq)

    def _decode_autoregressive(self, z, temperature=1.0, top_k=0, greedy=False):
        batch = tf.shape(z)[0]
        init_h = self.latent_to_init_h(z)
        z_context = self.latent_to_context(z)[:, tf.newaxis, :]

        token = tf.fill([batch, 1], START_ID)
        state = init_h
        all_logits = []

        for t in range(self.max_len):
            x = self.embedding(token)
            pos = self.pos_embedding(tf.fill([batch, 1], t))
            x = x + pos + z_context

            out, state = self.decoder_gru(x, initial_state=state, training=False)
            out = self.decoder_norm(out)
            step_logits = self.output_dense(out)
            all_logits.append(step_logits)

            sampling_logits = step_logits
            if temperature != 1.0:
                sampling_logits = sampling_logits / tf.maximum(temperature, 1e-6)
            if top_k and top_k > 0:
                k = tf.minimum(top_k, self.vocab_size)
                topk = tf.math.top_k(sampling_logits[:, 0, :], k=k)
                min_topk = topk.values[:, -1][:, tf.newaxis]
                mask = tf.cast(sampling_logits[:, 0, :] < min_topk, sampling_logits.dtype)
                filtered = sampling_logits[:, 0, :] + mask * -1e9
                sampling_logits = filtered[:, tf.newaxis, :]

            if greedy:
                token = tf.cast(tf.argmax(sampling_logits, axis=-1), tf.int32)
            else:
                sampled = tf.random.categorical(sampling_logits[:, 0, :], num_samples=1)
                token = tf.cast(sampled, tf.int32)

        return tf.concat(all_logits, axis=1)

    def decode(
        self,
        z,
        target_ids=None,
        training=False,
        temperature=1.0,
        top_k=0,
        greedy=False,
    ):
        if target_ids is not None:
            return self._decode_teacher_forcing(z, target_ids, training=training)
        return self._decode_autoregressive(
            z, temperature=temperature, top_k=top_k, greedy=greedy
        )

    def call(self, x, training=False):
        mean, log_var = self.encode(x, training=training)
        z = self.reparameterize(mean, log_var)
        logits = self.decode(z, target_ids=x, training=training)
        return logits, mean, log_var


def vae_loss(x, logits, mean, log_var, beta=0.05, kl_capacity=0.0):
    token_loss = tf.keras.losses.sparse_categorical_crossentropy(
        x, logits, from_logits=True
    )
    mask = tf.cast(tf.not_equal(x, PAD_ID), tf.float32)
    recon = tf.reduce_sum(token_loss * mask) / (tf.reduce_sum(mask) + 1e-6)

    kl = -0.5 * tf.reduce_mean(
        tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var), axis=1)
    )
    kl_penalty = tf.abs(kl - kl_capacity)
    return recon + beta * kl_penalty, recon, kl


# ---- Pretrained protein-vae adapter ----
import numpy as np
import torch


_PROTEIN_VAE_SEQ_LEN = 140
_PROTEIN_VAE_SEQ_CHOICES = [
    "G", "A", "L", "M", "F", "W", "K", "Q", "E", "S",
    "P", "V", "I", "C", "Y", "H", "R", "N", "D", "T", "X", "-",
]
_PROTEIN_VAE_N_SYMBOLS = len(_PROTEIN_VAE_SEQ_CHOICES)
_PROTEIN_VAE_INPUT_SIZE = 3088
_PROTEIN_VAE_HIDDEN_SIZES = [512, 256, 128, 16]
_PROTEIN_VAE_CONDITION_DIM = 8

_CILANTRO_AA = "ACDEFGHIKLMNPQRSTVWY"
_ID_TO_AA = {i + 3: aa for i, aa in enumerate(_CILANTRO_AA)}
_AA_TO_ID = {aa: i + 3 for i, aa in enumerate(_CILANTRO_AA)}


class _ProteinVAEBackbone(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super().__init__()
        self.fc = torch.nn.Linear(input_size, hidden_sizes[0])
        self.BN = torch.nn.BatchNorm1d(hidden_sizes[0])
        self.fc1 = torch.nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.BN1 = torch.nn.BatchNorm1d(hidden_sizes[1])
        self.fc2 = torch.nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.BN2 = torch.nn.BatchNorm1d(hidden_sizes[2])
        self.fc3_mu = torch.nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.fc3_sig = torch.nn.Linear(hidden_sizes[2], hidden_sizes[3])

        self.fc4 = torch.nn.Linear(hidden_sizes[3] + _PROTEIN_VAE_CONDITION_DIM, hidden_sizes[2])
        self.BN4 = torch.nn.BatchNorm1d(hidden_sizes[2])
        self.fc5 = torch.nn.Linear(hidden_sizes[2], hidden_sizes[1])
        self.BN5 = torch.nn.BatchNorm1d(hidden_sizes[1])
        self.fc6 = torch.nn.Linear(hidden_sizes[1], hidden_sizes[0])
        self.BN6 = torch.nn.BatchNorm1d(hidden_sizes[0])
        self.fc7 = torch.nn.Linear(hidden_sizes[0], input_size - _PROTEIN_VAE_CONDITION_DIM)

    def encode(self, x, code):
        h = torch.cat((x, code), dim=1)
        h = torch.relu(self.BN(self.fc(h)))
        h = torch.relu(self.BN1(self.fc1(h)))
        h = torch.relu(self.BN2(self.fc2(h)))
        mu = self.fc3_mu(h)
        sig = torch.nn.functional.softplus(self.fc3_sig(h))
        return mu, sig

    def decode_from_latent(self, z, code):
        h = torch.cat((z, code), dim=1)
        h = torch.relu(self.BN4(self.fc4(h)))
        h = torch.relu(self.BN5(self.fc5(h)))
        h = torch.relu(self.BN6(self.fc6(h)))
        return torch.sigmoid(self.fc7(h))


class ProteinSeqVAE:
    """Adapter exposing encode/decode using pretrained /protein-vae weights."""

    def __init__(self, weights_path, device="cpu"):
        self.device = torch.device(device)
        self.model = _ProteinVAEBackbone(_PROTEIN_VAE_INPUT_SIZE, _PROTEIN_VAE_HIDDEN_SIZES).to(self.device)
        state = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        self.latent_dim = _PROTEIN_VAE_HIDDEN_SIZES[-1]

    def _token_ids_to_seq(self, token_ids):
        chars = []
        for token in token_ids:
            token = int(token)
            if token in (PAD_ID, START_ID):
                continue
            aa = _ID_TO_AA.get(token)
            if aa:
                chars.append(aa)
        return "".join(chars)

    def _seq_to_input_vector(self, seq):
        seq = seq[:_PROTEIN_VAE_SEQ_LEN]
        gap_idx = _PROTEIN_VAE_SEQ_CHOICES.index("-")
        unknown_idx = _PROTEIN_VAE_SEQ_CHOICES.index("X")
        seq_indices = [gap_idx] * _PROTEIN_VAE_SEQ_LEN
        for idx, aa in enumerate(seq):
            seq_indices[idx] = _PROTEIN_VAE_SEQ_CHOICES.index(aa) if aa in _PROTEIN_VAE_SEQ_CHOICES else unknown_idx

        vec = np.zeros(_PROTEIN_VAE_SEQ_LEN * _PROTEIN_VAE_N_SYMBOLS, dtype=np.float32)
        for idx, aa_idx in enumerate(seq_indices):
            vec[idx * _PROTEIN_VAE_N_SYMBOLS + aa_idx] = 1.0
        return vec

    def _decoded_vec_to_token_ids(self, decoded_vec):
        aa_logits = decoded_vec.reshape(_PROTEIN_VAE_SEQ_LEN, _PROTEIN_VAE_N_SYMBOLS)
        token_ids = np.zeros(_PROTEIN_VAE_SEQ_LEN, dtype=np.int32)
        for idx in range(_PROTEIN_VAE_SEQ_LEN):
            aa = _PROTEIN_VAE_SEQ_CHOICES[int(np.argmax(aa_logits[idx]))]
            token_ids[idx] = _AA_TO_ID.get(aa, PAD_ID)
        return token_ids

    def encode(self, token_batch):
        token_np = np.asarray(token_batch)
        seq_vecs = np.stack([
            self._seq_to_input_vector(self._token_ids_to_seq(row)) for row in token_np
        ]).astype(np.float32)
        code = np.zeros((seq_vecs.shape[0], _PROTEIN_VAE_CONDITION_DIM), dtype=np.float32)

        with torch.no_grad():
            x = torch.from_numpy(seq_vecs).to(self.device)
            c = torch.from_numpy(code).to(self.device)
            mean, sig = self.model.encode(x, c)

        mean_np = mean.cpu().numpy()
        log_var_np = np.log(np.square(sig.cpu().numpy()) + 1e-8)
        return mean_np, log_var_np

    def decode(self, latent_batch):
        latent_np = np.asarray(latent_batch, dtype=np.float32)
        code = np.zeros((latent_np.shape[0], _PROTEIN_VAE_CONDITION_DIM), dtype=np.float32)

        with torch.no_grad():
            z = torch.from_numpy(latent_np).to(self.device)
            c = torch.from_numpy(code).to(self.device)
            decoded = self.model.decode_from_latent(z, c).cpu().numpy()

        return np.stack([self._decoded_vec_to_token_ids(row) for row in decoded])
