# data/ncbi_downloader.py
from Bio import Entrez, SeqIO
import time
import os

Entrez.email = "your.email@example.com"  # set your email

def fetch_sequences_by_taxon(taxon, out_fasta, max_records=500):
    """Fetch sequences from NCBI Nucleotide by taxon name (virus) - simple wrapper."""
    query = f"{taxon}[Organism] AND complete genome[Title]"
    handle = Entrez.esearch(db="nucleotide", term=query, retmax=max_records)
    record = Entrez.read(handle)
    ids = record["IdList"]
    if not ids:
        raise RuntimeError("No sequences found for query.")
    with open(out_fasta, "w") as out:
        for seq_id in ids:
            efetch = Entrez.efetch(db="nucleotide", id=seq_id, rettype="fasta", retmode="text")
            out.write(efetch.read())
            time.sleep(0.4)
    print(f"Wrote {out_fasta}")

if __name__ == "__main__":
    fetch_sequences_by_taxon("Influenza A virus", "data/influenza_a_sequences.fasta", max_records=200)
