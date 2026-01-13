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
