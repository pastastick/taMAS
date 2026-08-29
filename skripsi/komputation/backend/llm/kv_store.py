"""kv_store.py — persistensi KV-cache ke disk (file .pt + indeks SQLite).

Diekstrak apa adanya dari bekas `client.py` BAGIAN 3. Dipakai untuk resume
evolution loop / debugging mendalam — bukan jalur hot-path pipeline (yang
mengoper KV in-memory lewat `mas.kv_ops`). Terpisah dari `debug_log.py`: modul
itu menyimpan riwayat percakapan (input/output/hidden per langkah) untuk
decode-ulang manusiawi; modul ini menyimpan KV-cache itu sendiri.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from llm._shared import KVCache, _kv_to_cpu, _kv_to_device, _past_length


class KVCacheStore:
    """
    Penyimpanan KV-cache ke disk dalam dua tier:

    Tier FULL  (.pt di full/):
        Seluruh KV-cache disimpan.
        Dipakai untuk resume evolution loop atau debugging mendalam.

    Tier SELECTIVE (.pt di selective/):
        Subset token tertentu dari KV-cache.
        Saat ini: N token terakhir (token akhir paling informatif karena
        attention sudah mengakumulasi seluruh konteks sebelumnya).
        Placeholder untuk strategi KNN di masa depan.

    SQLite hanya menyimpan metadata (path, seq_len, ukuran, timestamp).
    Pencarian dan listing cepat tanpa harus load file .pt.

    Struktur direktori:
        kv_store/
          full/
            {conv_id}_{step:03d}.pt
          selective/
            {conv_id}_{step:03d}_sel.pt
          kv_index.db
    """

    def __init__(self, store_dir: str = "./debug/kv_store") -> None:
        self.store_dir = Path(store_dir)
        self.full_dir  = self.store_dir / "full"
        self.sel_dir   = self.store_dir / "selective"
        self.full_dir.mkdir(parents=True, exist_ok=True)
        self.sel_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        self.conn = sqlite3.connect(
            str(self.store_dir / "kv_index.db"), timeout=20
        )
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS kv_index (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                conv_id    TEXT    NOT NULL,
                step       INTEGER NOT NULL,
                tier       TEXT    NOT NULL,
                filepath   TEXT    NOT NULL,
                seq_len    INTEGER,
                n_layers   INTEGER,
                hidden_dim INTEGER,
                size_bytes INTEGER,
                created_at REAL,
                metadata   TEXT
            )
        """)
        self.conn.commit()

    def _index_add(self, conv_id: str, step: int, tier: str,
                   filepath: Path, kv: KVCache, metadata: Dict) -> None:
        from llm._shared import _is_dynamic_cache, _kv_pairs
        seq_len    = _past_length(kv)
        kv_pairs   = _kv_pairs(kv) if kv else []
        n_layers   = len(kv_pairs)
        hidden_dim = kv_pairs[0][0].shape[-1] if kv_pairs else 0
        size_bytes = filepath.stat().st_size if filepath.exists() else 0
        self.conn.execute(
            """INSERT INTO kv_index
               (conv_id,step,tier,filepath,seq_len,n_layers,
                hidden_dim,size_bytes,created_at,metadata)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (conv_id, step, tier, str(filepath), seq_len, n_layers,
             hidden_dim, size_bytes, time.time(), json.dumps(metadata))
        )
        self.conn.commit()

    def lookup(self, conv_id: str, step: int,
               tier: str = "full") -> Optional[Path]:
        """Cari path file KV-cache. Return None jika tidak ada."""
        cur = self.conn.execute(
            "SELECT filepath FROM kv_index "
            "WHERE conv_id=? AND step=? AND tier=? "
            "ORDER BY id DESC LIMIT 1",
            (conv_id, step, tier)
        )
        row = cur.fetchone()
        if row:
            p = Path(row[0])
            return p if p.exists() else None
        return None

    def list_entries(self, conv_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List semua entry (opsional filter by conv_id)."""
        if conv_id:
            cur = self.conn.execute(
                "SELECT conv_id,step,tier,seq_len,size_bytes,created_at "
                "FROM kv_index WHERE conv_id=? ORDER BY step",
                (conv_id,)
            )
        else:
            cur = self.conn.execute(
                "SELECT conv_id,step,tier,seq_len,size_bytes,created_at "
                "FROM kv_index ORDER BY created_at DESC LIMIT 100"
            )
        cols = ["conv_id","step","tier","seq_len","size_bytes","created_at"]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def save_full(self, conv_id: str, step: int, kv: KVCache,
                  metadata: Optional[Dict] = None) -> Path:
        """Simpan full KV-cache (dipindah ke CPU sebelum simpan)."""
        metadata = metadata or {}
        filepath = self.full_dir / f"{conv_id}_{step:03d}.pt"
        torch.save(_kv_to_cpu(kv), filepath)
        self._index_add(conv_id, step, "full", filepath, kv, metadata)
        return filepath

    def save_selective(self, conv_id: str, step: int, kv: KVCache,
                       n_tokens: int = 64,
                       metadata: Optional[Dict] = None) -> Path:
        """
        Simpan N token terakhir dari KV-cache.

        KV-cache shape per layer: [batch, n_heads, seq_len, head_dim]
        Slice pada dim=-2 (seq_len): kv[..., -n_tokens:, :]

        Kenapa token terakhir?
          Dalam transformer causal, token posisi akhir dalam KV-cache
          sudah melalui attention dengan semua token sebelumnya.
          Konteks terkini paling relevan untuk prediksi berikutnya
          tersimpan di sini.
        """
        from llm._shared import _kv_pairs
        metadata = metadata or {}
        seq_len  = _past_length(kv)
        keep     = min(n_tokens, seq_len)

        selective: KVCache = tuple(
            (k[..., -keep:, :].cpu(), v[..., -keep:, :].cpu())
            for k, v in _kv_pairs(kv)
        )

        filepath = self.sel_dir / f"{conv_id}_{step:03d}_sel.pt"
        torch.save(selective, filepath)
        self._index_add(conv_id, step, "selective", filepath, selective,
                        {**metadata, "n_tokens": keep, "original_seq_len": seq_len})
        return filepath

    def load(self, conv_id: str, step: int, tier: str = "full",
             device: Optional[torch.device] = None) -> Optional[KVCache]:
        """Load KV-cache. Opsional pindah ke device setelah load."""
        filepath = self.lookup(conv_id, step, tier)
        if filepath is None:
            return None
        kv = torch.load(filepath, map_location="cpu", weights_only=True)
        if device is not None:
            kv = _kv_to_device(kv, device)
        return kv

    def delete(self, conv_id: str, step: int, tier: str = "full") -> bool:
        filepath = self.lookup(conv_id, step, tier)
        if filepath and filepath.exists():
            filepath.unlink()
        self.conn.execute(
            "DELETE FROM kv_index WHERE conv_id=? AND step=? AND tier=?",
            (conv_id, step, tier)
        )
        self.conn.commit()
        return filepath is not None

    def total_size_mb(self) -> float:
        cur = self.conn.execute("SELECT SUM(size_bytes) FROM kv_index")
        total = cur.fetchone()[0] or 0
        return total / (1024 ** 2)

    # ── KV-cache pruning ──────────────────────────────────────────────────

    def prune_by_age(self, max_age_seconds: float) -> int:
        """
        Delete KV-cache entries older than max_age_seconds.
        Returns number of entries deleted.
        """
        cutoff = time.time() - max_age_seconds
        cur = self.conn.execute(
            "SELECT id, filepath FROM kv_index WHERE created_at < ?",
            (cutoff,)
        )
        rows = cur.fetchall()
        for row_id, fpath in rows:
            p = Path(fpath)
            if p.exists():
                p.unlink()
        self.conn.execute(
            "DELETE FROM kv_index WHERE created_at < ?", (cutoff,)
        )
        self.conn.commit()
        return len(rows)

    def prune_by_size(self, max_total_mb: float) -> int:
        """
        Evict oldest entries until total disk usage is under max_total_mb.
        Returns number of entries deleted.
        """
        deleted = 0
        while self.total_size_mb() > max_total_mb:
            cur = self.conn.execute(
                "SELECT id, filepath FROM kv_index "
                "ORDER BY created_at ASC LIMIT 1"
            )
            row = cur.fetchone()
            if row is None:
                break
            row_id, fpath = row
            p = Path(fpath)
            if p.exists():
                p.unlink()
            self.conn.execute("DELETE FROM kv_index WHERE id=?", (row_id,))
            self.conn.commit()
            deleted += 1
        return deleted

    def prune_by_count(self, max_entries: int) -> int:
        """
        Keep only the most recent max_entries entries (per tier).
        Deletes oldest entries first. Returns number of entries deleted.
        """
        deleted = 0
        for tier in ("full", "selective"):
            cur = self.conn.execute(
                "SELECT COUNT(*) FROM kv_index WHERE tier=?", (tier,)
            )
            count = cur.fetchone()[0]
            excess = count - max_entries
            if excess <= 0:
                continue
            cur = self.conn.execute(
                "SELECT id, filepath FROM kv_index "
                "WHERE tier=? ORDER BY created_at ASC LIMIT ?",
                (tier, excess)
            )
            rows = cur.fetchall()
            for row_id, fpath in rows:
                p = Path(fpath)
                if p.exists():
                    p.unlink()
                self.conn.execute("DELETE FROM kv_index WHERE id=?", (row_id,))
            self.conn.commit()
            deleted += len(rows)
        return deleted

    def prune_keep_conv_ids(self, keep_conv_ids: List[str]) -> int:
        """
        Delete all entries EXCEPT those matching keep_conv_ids.
        Useful after evolution round: keep only best trajectories' KV-caches.
        Returns number of entries deleted.
        """
        if not keep_conv_ids:
            return 0
        placeholders = ",".join("?" for _ in keep_conv_ids)
        cur = self.conn.execute(
            f"SELECT id, filepath FROM kv_index "
            f"WHERE conv_id NOT IN ({placeholders})",
            keep_conv_ids,
        )
        rows = cur.fetchall()
        for row_id, fpath in rows:
            p = Path(fpath)
            if p.exists():
                p.unlink()
        self.conn.execute(
            f"DELETE FROM kv_index WHERE conv_id NOT IN ({placeholders})",
            keep_conv_ids,
        )
        self.conn.commit()
        return len(rows)

    def auto_prune(
        self,
        max_total_mb: float = 2048.0,
        max_age_seconds: Optional[float] = None,
        max_entries_per_tier: Optional[int] = None,
    ) -> Dict[str, int]:
        """
        Run all applicable pruning strategies. Called automatically after save
        if auto_prune is enabled on the store.

        Returns dict with counts of deleted entries per strategy.
        """
        result: Dict[str, int] = {}
        if max_age_seconds is not None:
            result["age"] = self.prune_by_age(max_age_seconds)
        if max_entries_per_tier is not None:
            result["count"] = self.prune_by_count(max_entries_per_tier)
        # Size-based pruning always runs as final guard
        result["size"] = self.prune_by_size(max_total_mb)
        return result
