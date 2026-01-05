# train.py
import argparse
import tensorflow as tf
import numpy as np
from models.vae import SeqVAE, vae_loss
from utils.seq_utils import build_vocab_and_encode, load_fasta_as_sequences
from tqdm import tqdm
import os
import json

def train(args):
    seqs = load_fasta_as_sequences(args.input_fasta)
    tokenized, vocab = build_vocab_and_encode(seqs, max_len=args.max_len)
    dataset = tf.data.Dataset.from_tensor_slices(tokenized)
    dataset = dataset.shuffle(1000).batch(args.batch_size, drop_remainder=True)
    model = SeqVAE(vocab_size=len(vocab), emb_dim=args.emb_dim, enc_units=args.enc_units,
                   latent_dim=args.latent_dim, dec_units=args.dec_units, max_len=args.max_len)
    opt = tf.keras.optimizers.Adam(args.lr)
    ckpt_dir = args.ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    @tf.function
    def train_step(x):
        with tf.GradientTape() as tape:
            logits, z_mean, z_log_var = model(x, training=True)
            loss, recon, kl = vae_loss(x, logits, z_mean, z_log_var)
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss, recon, kl

    for epoch in range(args.epochs):
        pbar = tqdm(dataset, desc=f"Epoch {epoch}")
        for step, batch in enumerate(pbar):
            loss, recon, kl = train_step(batch)
            if step % 50 == 0:
                pbar.set_postfix({"loss": float(loss), "recon": float(recon), "kl": float(kl)})
        model.save_weights(os.path.join(ckpt_dir, f"vae_epoch{epoch}.ckpt"))
        meta = {"epoch": epoch, "args": vars(args)}
        with open(os.path.join(ckpt_dir, "metadata.json"), "w") as f:
            json.dump(meta, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_fasta", default="cilantro/data/example_sequences.fasta")
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--enc_units", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--dec_units", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ckpt_dir", default="checkpoints")
    args = parser.parse_args()
    train(args)
