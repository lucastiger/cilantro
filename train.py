# train.py
import argparse
import tensorflow as tf
from models.vae import SeqVAE, vae_loss
from utils.seq_utils import build_vocab_and_encode, load_fasta_as_sequences
from tqdm import tqdm
import os
import json


def compute_kl_control(step, total_steps, beta_min, beta_max, warmup_frac, capacity_max):
    warmup_steps = max(int(total_steps * warmup_frac), 1)
    progress = min(step / warmup_steps, 1.0)
    beta = beta_min + (beta_max - beta_min) * progress
    capacity = capacity_max * progress
    return beta, capacity

def train(args):
    seqs = load_fasta_as_sequences(args.input_fasta)
    tokenized, vocab = build_vocab_and_encode(seqs, max_len=args.max_len)
    dataset = tf.data.Dataset.from_tensor_slices(tokenized)
    dataset = dataset.shuffle(1000).batch(args.batch_size, drop_remainder=True)

    #add 3 for PAD, UNK, START tokens
    model = SeqVAE(
        vocab_size=len(vocab) + 3,
        emb_dim=args.emb_dim,
        enc_units=args.enc_units,
        latent_dim=args.latent_dim,
        dec_units=args.dec_units,
        max_len=args.max_len,
        dropout=args.dropout,
        gru_dropout=args.gru_dropout,
        pooling_heads=args.pooling_heads,
    )
    model.build((None, args.max_len))
    
    opt = tf.keras.optimizers.Adam(args.lr)
    ckpt_dir = args.ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    steps_per_epoch = len(tokenized) // args.batch_size
    total_steps = max(steps_per_epoch * args.epochs, 1)

    @tf.function
    def train_step(x, beta, capacity):
        with tf.GradientTape() as tape:
            logits, z_mean, z_log_var = model(x, training=True)
            loss, recon, kl = vae_loss(
                x,
                logits,
                z_mean,
                z_log_var,
                beta=beta,
                kl_capacity=capacity,
            )
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss, recon, kl

    global_step = 0
    for epoch in range(args.epochs):
        pbar = tqdm(dataset, desc=f"Epoch {epoch}")
        for step, batch in enumerate(pbar):
            beta, capacity = compute_kl_control(
                global_step,
                total_steps,
                args.kl_beta_min,
                args.kl_beta,
                args.kl_warmup_frac,
                args.kl_capacity_max,
            )
            loss, recon, kl = train_step(
                batch,
                beta=tf.constant(beta, dtype=tf.float32),
                capacity=tf.constant(capacity, dtype=tf.float32),
            )
            if step % 50 == 0:
                pbar.set_postfix(
                    {
                        "loss": float(loss),
                        "recon": float(recon),
                        "kl": float(kl),
                        "beta": beta,
                        "cap": capacity,
                    }
                )
            global_step += 1
        model.save_weights(os.path.join(ckpt_dir, f"vae_epoch{epoch}.weights.h5"))
        meta = {"epoch": epoch, "args": vars(args)}
        with open(os.path.join(ckpt_dir, "metadata.json"), "w") as f:
            json.dump(meta, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_fasta", default="cilantro/data/example_sequences.fasta")
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--enc_units", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--dec_units", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--gru_dropout", type=float, default=0.1)
    parser.add_argument("--pooling_heads", type=int, default=4)
    parser.add_argument("--kl_beta", type=float, default=0.05)
    parser.add_argument("--kl_beta_min", type=float, default=1e-3)
    parser.add_argument("--kl_warmup_frac", type=float, default=0.3)
    parser.add_argument("--kl_capacity_max", type=float, default=8.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ckpt_dir", default="checkpoints")
    args = parser.parse_args()
    train(args)
