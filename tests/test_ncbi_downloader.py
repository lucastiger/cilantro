import http.client

import pytest

from data import ncbi_downloader


class _FakeHandle:
    def __init__(self, payload: str, fail_once: bool = False):
        self.payload = payload
        self._fail_once = fail_once

    def read(self) -> str:
        if self._fail_once:
            self._fail_once = False
            raise http.client.IncompleteRead(b"partial")
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_fetch_batch_retries_incomplete_read(monkeypatch):
    calls = {"count": 0}

    def fake_efetch(**kwargs):
        calls["count"] += 1
        return _FakeHandle(">id\nAAA\n", fail_once=(calls["count"] == 1)
        )

    monkeypatch.setattr(ncbi_downloader.Entrez, "efetch", fake_efetch)

    chunk = ncbi_downloader._fetch_fasta_batch_with_retries(
        db="protein",
        batch=["1", "2"],
        max_retries=3,
        retry_backoff_s=0,
    )

    assert chunk == ">id\nAAA\n"
    assert calls["count"] == 2


def test_fetch_batch_raises_after_exhausted_retries(monkeypatch):
    def always_fail(**kwargs):
        return _FakeHandle("", fail_once=True)

    monkeypatch.setattr(ncbi_downloader.Entrez, "efetch", always_fail)

    with pytest.raises(RuntimeError, match="Failed to fetch FASTA batch"):
        ncbi_downloader._fetch_fasta_batch_with_retries(
            db="protein",
            batch=["1"],
            max_retries=2,
            retry_backoff_s=0,
        )
