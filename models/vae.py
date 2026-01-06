import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class SeqVAE(Model):
    def __init__(self, vocab_size, emb_dim=64, enc_units=256, latent_dim=64, dec_units=256, max_len=200):
        super().__init__()
        self.max_len = max_len
        self.latent_dim = latent_dim

        self.embedding = layers.Embedding(vocab_size, emb_dim, mask_zero=True)
        
        self.encoder = layers.Bidirectional(layers.LSTM(enc_units, return_sequences=False))
        
        self.z_mean = layers.Dense(latent_dim)
        self.z_log_var = layers.Dense(latent_dim)
        
        self.latent_to_hidden = layers.Dense(dec_units, activation="tanh")
        
        self.decoder_cell = layers.LSTMCell(dec_units)
        self.decoder_dense = layers.Dense(vocab_size)

    # Build method to declare input shape to Keras
    def build(self, input_shape):
        super().build(input_shape)
    
    def encode(self, x):
        h = self.encoder(self.embedding(x))
        return self.z_mean(h), self.z_log_var(h)

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(0.5 * logvar) + mean


    def decode(self, z):
        batch = tf.shape(z)[0]
        h = self.latent_to_hidden(z)
        c = tf.zeros_like(h)

        outputs = []
        token = tf.fill([batch], 2)  # START token

        for _ in range(self.max_len):
            emb = self.embedding(token)
            h, c = self.decoder_cell(emb, [h, c])
            logits = self.decoder_dense(h)
            outputs.append(logits)
            token = tf.argmax(logits, axis=-1)

        return tf.stack(outputs, axis=1)

    def call(self, x, training=False):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        logits = self.decode(z)
        return logits, z_mean, z_log_var

def vae_loss(x_true, logits, z_mean, z_log_var, pad_token=0):
    recon = tf.keras.losses.sparse_categorical_crossentropy(x_true, logits, from_logits=True)
    mask = tf.cast(tf.not_equal(x_true, pad_token), tf.float32)
    recon_loss = tf.reduce_sum(recon * mask) / tf.reduce_sum(mask)
    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    return recon_loss + 1e-3 * kl_loss, recon_loss, kl_loss
