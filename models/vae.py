import tensorflow as tf
from tensorflow.keras import layers, Model

class SeqVAE(Model):
    def __init__(
        self,
        vocab_size,
        emb_dim=64,
        enc_units=256,
        latent_dim=64,
        dec_units=256,
        max_len=200,
    ):
        super().__init__()
        self.max_len = max_len
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size

        # ---- Encoder ----
        self.embedding = layers.Embedding(
            vocab_size, emb_dim, mask_zero=True
        )
        self.encoder = layers.Bidirectional(
            layers.LSTM(enc_units, return_sequences=False)
        )

        self.z_mean = layers.Dense(latent_dim)
        self.z_log_var = layers.Dense(latent_dim)

        # ---- Decoder ----
        self.latent_to_init = layers.Dense(dec_units, activation="tanh")

        self.decoder_cell = layers.LSTMCell(dec_units)
        self.decoder_rnn = layers.RNN(
            self.decoder_cell, return_sequences=True
        )

        self.output_dense = layers.Dense(vocab_size)

    def encode(self, x):
        x = self.embedding(x)
        h = self.encoder(x)
        return self.z_mean(h), self.z_log_var(h)

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(tf.shape(mean))
        return mean + tf.exp(0.5 * logvar) * eps

    def decode(self, z):
        batch = tf.shape(z)[0]

        # Initial hidden state
        h0 = self.latent_to_init(z)
        c0 = tf.zeros_like(h0)

        # Decoder inputs: START token repeated
        start_tokens = tf.fill([batch, self.max_len], 2)
        emb = self.embedding(start_tokens)

        # Run RNN
        h_seq = self.decoder_rnn(
            emb, initial_state=[h0, c0]
        )

        logits = self.output_dense(h_seq)
        return logits

    def call(self, x, training=False):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        logits = self.decode(z)
        return logits, mean, log_var


def vae_loss(x, logits, mean, log_var):
    recon = tf.keras.losses.sparse_categorical_crossentropy(
        x, logits, from_logits=True
    )
    recon = tf.reduce_mean(recon)

    kl = -0.5 * tf.reduce_mean(
        1 + log_var - tf.square(mean) - tf.exp(log_var)
    )

    return recon + 1e-3 * kl, recon, kl
