# utils/seq_utils.py
import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
PAD_ID = 0
UNK_ID = 1
START_ID = 2

def build_vocab_and_encode(seqs, max_len):
    aa_to_id = {aa: i + 3 for i, aa in enumerate(AA)}
    encoded = []

    for s in seqs:
        ids = [START_ID]
        for ch in s[: max_len - 1]:
            ids.append(aa_to_id.get(ch, UNK_ID))
        ids += [PAD_ID] * (max_len - len(ids))
        encoded.append(ids)

    return encoded, aa_to_id

def decode_sequence_from_ids(ids):
    id_to_aa = {i + 3: aa for i, aa in enumerate(AA)}
    seq = ""
    for x in ids:
        if x <= 2:
            continue
        seq += id_to_aa.get(int(x), "")
    return seq

def load_fasta_as_sequences(path):
    seqs = []
    from Bio import SeqIO
    for rec in SeqIO.parse(path, "fasta"):
        seqs.append(str(rec.seq).upper())
    return seqs
