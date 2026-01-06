import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class SeqVAE(Model):
    def __init__(self, vocab_size, emb_dim=64, enc_units=256, latent_dim=64, dec_units=256, max_len=200):
        super().__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.embedding = layers.Embedding(vocab_size, emb_dim, mask_zero=True)
        self.encoder_lstm = layers.Bidirectional(layers.LSTM(enc_units, return_sequences=False))
        self.z_mean = layers.Dense(latent_dim)
        self.z_log_var = layers.Dense(latent_dim)
        self.latent_to_init = layers.Dense(dec_units, activation="tanh")
        self.decoder_lstm_cell = layers.LSTMCell(dec_units)
        self.decoder_rnn = layers.RNN(self.decoder_lstm_cell, return_sequences=True, return_state=True)
        self.decoder_dense = layers.Dense(vocab_size)

    def encode(self, x):
        x_emb = self.embedding(x)
        h = self.encoder_lstm(x_emb)
        z_mean = self.z_mean(h)
        z_log_var = self.z_log_var(h)
        return z_mean, z_log_var

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(0.5 * logvar) + mean

    @tf.function
    def decode(self, z, seq_len=None, training=False, teacher=None):
        if seq_len is None:
            seq_len = self.max_len
        batch = tf.shape(z)[0]
        init_state = [self.latent_to_init(z), tf.zeros_like(self.latent_to_init(z))]
        # start token id = 2, teacher forcing optional
        start_tokens = tf.fill([batch, 1], 2)
        emb = self.embedding(start_tokens)
        
        # Use TensorArray to collect outputs
        outputs_ta = tf.TensorArray(dtype=tf.float32, size=seq_len, dynamic_size=False, clear_after_read=False)
        
        state = init_state
        for t in tf.range(seq_len):
            out, h, c = self.decoder_rnn(emb, initial_state=state)
            logits = self.decoder_dense(out)
            outputs_ta = outputs_ta.write(t, logits) # Write to TensorArray
            if training and teacher is not None:
                # next input from teacher forcing
                next_id = tf.expand_dims(teacher[:, t], 1)
            else:
                next_id = tf.argmax(logits, axis=-1)
            emb = self.embedding(tf.cast(next_id, tf.int32))
            state = [h, c]
        
        # Stack the TensorArray to get a single tensor and reshape if necessary
        outputs = outputs_ta.stack()
        # The `outputs_ta.stack()` will create a tensor of shape (seq_len, batch_size, vocab_size).
        # We need to transpose it to (batch_size, seq_len, vocab_size).
        outputs = tf.transpose(outputs, perm=[1, 0, 2])
        
        return outputs

    def call(self, x, training=False):
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        logits = self.decode(z, seq_len=tf.shape(x)[1], training=training, teacher=x if training else None)
        return logits, z_mean, z_log_var

def vae_loss(x_true, logits, z_mean, z_log_var, pad_token=0):
    recon = tf.keras.losses.sparse_categorical_crossentropy(x_true, logits, from_logits=True)
    mask = tf.cast(tf.not_equal(x_true, pad_token), tf.float32)
    recon_loss = tf.reduce_sum(recon * mask) / tf.reduce_sum(mask)
    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    return recon_loss + 1e-3 * kl_loss, recon_loss, kl_loss
