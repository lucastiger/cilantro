# data/ncbi_downloader.py
import argparse
import os
import time
from typing import Iterable, List

from Bio import Entrez


def _set_entrez_credentials(email: str | None, api_key: str | None) -> None:
    if not email:
        raise ValueError(
            "NCBI requires an email address. Pass --email or set NCBI_EMAIL."
        )
    Entrez.email = email
    if api_key:
        Entrez.api_key = api_key


def _chunked(ids: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(ids), batch_size):
        yield ids[i : i + batch_size]


def fetch_sequences_by_taxon(
    taxon: str,
    out_fasta: str,
    max_records: int = 500,
    db: str = "protein",
    query: str | None = None,
    batch_size: int = 50,
    sleep_s: float | None = None,
) -> None:
    search_query = query or f"{taxon}[Organism]"
    handle = Entrez.esearch(db=db, term=search_query, retmax=max_records)
    record = Entrez.read(handle)
    ids = record["IdList"]

    if not ids:
        raise RuntimeError(f"No sequences found for query: {search_query}")

    if sleep_s is None:
        sleep_s = 0.11 if getattr(Entrez, "api_key", None) else 0.34

    with open(out_fasta, "w") as out:
        for batch in _chunked(ids, batch_size):
            efetch = Entrez.efetch(
                db=db,
                id=",".join(batch),
                rettype="fasta",
                retmode="text",
            )
            out.write(efetch.read())
            time.sleep(sleep_s)

    print(f"Wrote {out_fasta}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--taxon", required=True)
    parser.add_argument("--out_fasta", required=True)
    parser.add_argument("--max_records", type=int, default=500)
    parser.add_argument("--db", default="protein")
    parser.add_argument("--query", default=None)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--sleep_s", type=float, default=None)
    parser.add_argument("--email", default=os.getenv("NCBI_EMAIL"))
    parser.add_argument("--api_key", default=os.getenv("NCBI_API_KEY"))
    args = parser.parse_args()

    _set_entrez_credentials(args.email, args.api_key)
    fetch_sequences_by_taxon(
        taxon=args.taxon,
        out_fasta=args.out_fasta,
        max_records=args.max_records,
        db=args.db,
        query=args.query,
        batch_size=args.batch_size,
        sleep_s=args.sleep_s,
    )


if __name__ == "__main__":
    main()
