"""backend.py — LocalLLMBackend, API publik utama paket llm/.

Diekstrak dari bekas `client.py` BAGIAN 7. Membungkus `engine._CoreEngine`
dengan lock, snapshot debug (`debug/llm_outputs/*.md`), persistensi opsional
(`kv_store.KVCacheStore`, `debug_log.TensorConvManager`), dan kompatibilitas
API lama (`build_messages_and_create_chat_completion`, dst).

Catatan korektnes pemecahan file (bukan kosmetik — dibaca sebelum menyunting
`_save_output_snapshot` atau `__init__`): state `_GLOBAL_OUTPUT_*` di
`engine.py` DIMUTASI dari sini (counter panggilan LLM bersama lintas semua
instance `LocalLLMBackend`, di modul mana pun instance itu dibuat). `global X`
di dalam sebuah fungsi selalu menunjuk ke modul TEMPAT fungsi itu
didefinisikan — bukan tempat variabelnya aslinya dideklarasikan — jadi setelah
`_save_output_snapshot` pindah ke sini, `global _GLOBAL_CALL_COUNTER` tidak
lagi menunjuk variabel yang sama di `engine.py`. Diganti akses atribut modul
terkualifikasi (`engine._GLOBAL_CALL_COUNTER`), yang membaca/menulis nilai
hidup di `engine.py` apa pun urutan panggilan antar-modul/instance —
semantik counter bersama tetap identik dengan sebelum file dipecah.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from llm import engine
from llm._shared import KVCache, OutputMode, _past_length, kv_truncate
from llm.debug_log import ConvRecord, TensorConvManager
from llm.engine import LLMResult
from llm.kv_store import KVCacheStore
from llm.session import LocalChatSession


class LocalLLMBackend:
    """
    Backend LLM lokal sebagai pengganti APIBackend di

    Kompatibel dengan APIBackend (client.py):
        build_messages()
        build_messages_and_create_chat_completion()
        build_chat_session()

    Tambahan untuk latent reasoning:
        run()                 <- titik masuk tunggal, mode fleksibel
        run_latent_pass()     <- shortcut kv_only
        run_generate()        <- shortcut text_only atau kv_and_text
        load_kv()             <- resume dari KV-cache tersimpan
        debug_step()          <- decode tensor log untuk debugging

    Args:
        model_name      : nama model HuggingFace
        device          : "cuda", "cuda:0", "cpu"
        latent_steps    : jumlah latent step per agen (0 = nonaktif)
        use_realign     : aktifkan proyeksi realignment
        enable_thinking : aktifkan thinking Qwen3 (default False = hemat token)
        log_tensors     : simpan tensor ke TensorConvManager
        store_kv        : simpan KV-cache ke disk
        conv_dir        : direktori log tensor
        kv_dir          : direktori KV-cache
        max_new_tokens  : batas token output
        temperature     : sampling temperature
        top_p           : nucleus sampling
    """

    DEFAULT_SYSTEM_PROMPT = "You are a helpful quantitative finance AI assistant."

    def __init__(
        self,
        model_name     : str   = "Qwen/Qwen3-4B",
        device         : str   = "cuda",
        latent_steps   : int   = 0,
        use_realign    : bool  = False,
        enable_thinking: bool  = False,
        log_tensors    : bool  = False,
        store_kv       : bool  = False,
        conv_dir       : str   = "./debug/conv_logs",
        kv_dir         : str   = "./debug/kv_store",
        output_log_dir : str   = "./debug/llm_outputs",
        max_new_tokens : int   = 2048,
        temperature    : float = 0.6,
        top_p          : float = 0.95,
        kv_prune_max_mb        : float = 2048.0,
        kv_prune_max_age_secs  : Optional[float] = None,
        kv_prune_max_entries   : Optional[int]    = None,
        kv_max_seq_len         : Optional[int]    = None,
        # KNN-based KV-cache filtering
        knn_enabled    : bool  = False,
        knn_percentage : float = 0.8,
        knn_min_keep   : int   = 5,
        knn_strategy   : str   = "top",
        # Mode langkah laten (B2/G3). None = pakai env LATENT_STEP_MODE.
        latent_step_mode: Optional[str]   = None,
        latent_step_temp: Optional[float] = None,
        # β untuk mode "moi" (MoI arXiv:2505.14827). None = env LATENT_STEP_BETA.
        latent_step_beta: Optional[float] = None,
        # α untuk mode "mix" (sumbu interpolasi raw<->soft). None = env
        # LATENT_STEP_ALPHA.
        latent_step_alpha: Optional[float] = None,
        # Early-stop rollout laten (B6). None = pakai env LATENT_EARLY_STOP_COS
        # (default 0.999); ≥ 1.0 mematikan early-stop.
        latent_early_stop_cos: Optional[float] = None,
    ) -> None:

        self.max_new_tokens = max_new_tokens
        self.temperature    = temperature
        self.top_p          = top_p
        self.log_tensors    = log_tensors
        self.store_kv       = store_kv

        # KV-cache pruning config
        self._kv_prune_max_mb       = kv_prune_max_mb
        self._kv_prune_max_age_secs = kv_prune_max_age_secs
        self._kv_prune_max_entries  = kv_prune_max_entries
        self._kv_max_seq_len        = kv_max_seq_len

        _device = torch.device(
            device if (torch.cuda.is_available() or device == "cpu") else "cpu"
        )

        self._lock     = threading.Lock()

        self._engine   = engine._CoreEngine(
            model_name=model_name, device=_device,
            latent_steps=latent_steps, use_realign=use_realign,
            enable_thinking=enable_thinking,
            knn_enabled=knn_enabled, knn_percentage=knn_percentage,
            knn_min_keep=knn_min_keep, knn_strategy=knn_strategy,
            latent_step_mode=latent_step_mode, latent_step_temp=latent_step_temp,
            latent_step_beta=latent_step_beta,
            latent_step_alpha=latent_step_alpha,
            latent_early_stop_cos=latent_early_stop_cos,
        )
        self._conv_mgr = TensorConvManager(conv_dir) if log_tensors else None
        self._kv_store = KVCacheStore(kv_dir)        if store_kv    else None

        # ── LLM output log: satu folder flat untuk seluruh proses ───────────
        # Semua instance berbagi _GLOBAL_OUTPUT_LOG_DIR (inisialisasi idempoten).
        # Tidak ada session subfolder — semua file ada di satu tempat.
        engine._init_global_output_dir(output_log_dir)
        self._output_log_dir = engine._GLOBAL_OUTPUT_LOG_DIR

    # ── Kompatibilitas APIBackend ──────────────────────────────────────────

    def build_messages(
        self,
        user_prompt    : str,
        system_prompt  : Optional[str]      = None,
        former_messages: Optional[List[Dict]] = None,
    ) -> List[Dict[str, str]]:
        """Susun messages [system, history, user]. Identik dengan APIBackend."""
        msgs = [{"role": "system",
                 "content": system_prompt or self.DEFAULT_SYSTEM_PROMPT}]
        if former_messages:
            msgs.extend(former_messages)
        msgs.append({"role": "user", "content": user_prompt})
        return msgs

    def build_messages_and_create_chat_completion(
        self,
        user_prompt    : str,
        system_prompt  : Optional[str]      = None,
        former_messages: Optional[List[Dict]] = None,
        *,
        json_mode      : bool = False,
        past_key_values: Optional[KVCache]  = None,
        max_new_tokens : Optional[int]      = None,
        temperature    : Optional[float]    = None,
        top_p          : Optional[float]    = None,
        conv_id        : Optional[str]      = None,
        step           : int = 0,
        role           : str = "assistant",
        mode           : Optional[OutputMode] = None,
        latent_steps   : Optional[int] = None,
        json_schema    : Optional[Dict[str, Any]] = None,
        prefill        : str = "",
    ) -> str:
        """
        Drop-in replacement untuk APIBackend.build_messages_and_create_chat_completion.

        Tambahan dibanding APIBackend:
          past_key_values : KV-cache dari agen sebelumnya
          conv_id/step/role : untuk logging TensorConvManager
          mode            : output mode override.  Default: "kv_and_text"
                            jika latent_steps > 0 atau past_key_values ada,
                            otherwise "text_only" (backward compatible).
        """
        if mode is None:
            # Auto-select: use kv_and_text when latent steps are configured
            # or external KV-cache is provided, otherwise text_only
            if self._engine.latent_steps > 0 or past_key_values is not None:
                mode = "kv_and_text"
            else:
                mode = "text_only"

        messages = self.build_messages(user_prompt, system_prompt, former_messages)
        result   = self.run(
            messages=messages, mode=mode,
            past_key_values=past_key_values,
            max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
            json_mode=json_mode, conv_id=conv_id, step=step, role=role,
            latent_steps=latent_steps, json_schema=json_schema,
            prefill=prefill,
        )
        return result.text or ""

    def build_messages_and_run(
        self,
        user_prompt    : str,
        system_prompt  : Optional[str]      = None,
        former_messages: Optional[List[Dict]] = None,
        *,
        json_mode      : bool = False,
        past_key_values: Optional[KVCache]  = None,
        max_new_tokens : Optional[int]      = None,
        temperature    : Optional[float]    = None,
        top_p          : Optional[float]    = None,
        conv_id        : Optional[str]      = None,
        step           : int = 0,
        role           : str = "assistant",
        mode           : Optional[OutputMode] = None,
        latent_steps   : Optional[int] = None,
        json_schema    : Optional[Dict[str, Any]] = None,
        crop_after_generate: bool = True,
        prefill        : str = "",
    ) -> LLMResult:
        """
        Sama seperti build_messages_and_create_chat_completion, tapi return
        LLMResult lengkap (termasuk kv_cache, hidden_last, latent_vecs).

        crop_after_generate : bila True (default infra), buang token jawaban dari
            KV setelah generate (anti-contamination). Set False untuk NO-CROP
            (pipeline produksi) agar jawaban asli ikut di-chain ke agent berikut.

        Dipakai oleh Latent pipeline classes (LatentHypothesisGen, dsb.)
        yang butuh akses ke KV-cache output untuk di-chain ke step berikutnya.
        """
        if mode is None:
            if self._engine.latent_steps > 0 or past_key_values is not None:
                mode = "kv_and_text"
            else:
                mode = "text_only"

        messages = self.build_messages(user_prompt, system_prompt, former_messages)
        return self.run(
            messages=messages, mode=mode,
            past_key_values=past_key_values,
            max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
            json_mode=json_mode, conv_id=conv_id, step=step, role=role,
            latent_steps=latent_steps, json_schema=json_schema,
            crop_after_generate=crop_after_generate,
            prefill=prefill,
        )

    # ── Titik masuk fleksibel ──────────────────────────────────────────────

    def run(
        self,
        messages           : List[Dict[str, str]],
        mode               : OutputMode = "text_only",
        past_key_values    : Optional[KVCache]  = None,
        max_new_tokens     : Optional[int]      = None,
        temperature        : Optional[float]    = None,
        top_p              : Optional[float]    = None,
        json_mode          : bool = False,
        record_latent_vecs : bool = True,
        conv_id            : Optional[str] = None,
        step               : int  = 0,
        role               : str  = "agent",
        kv_n_selective     : int  = 64,
        latent_steps       : Optional[int] = None,
        json_schema        : Optional[Dict[str, Any]] = None,
        crop_after_generate: bool = True,
        prefill            : str = "",
    ) -> LLMResult:
        """
        Titik masuk tunggal untuk semua mode.

        mode="kv_only":
            Jalankan latent pass, kembalikan KV-cache.
            Gunakan untuk: propose, construct — agen yang hanya membangun konteks.
            text=None, kv_cache ada.

        mode="text_only":
            Generate teks, tidak kembalikan KV-cache.
            Gunakan untuk: judger, feedback — agen yang butuh output string.
            text ada, kv_cache=None.

        mode="kv_and_text":
            Latent pass dulu, lalu generate teks dari KV yang dibangun.
            Gunakan untuk: agen yang butuh KEDUANYA.
            text ada, kv_cache ada.

        Args:
            messages          : list message
            mode              : output mode
            past_key_values   : KV-cache dari agen sebelumnya
            json_mode         : auto-extract + fix JSON dari output
            record_latent_vecs: catat latent vectors ke log
            conv_id           : ID percakapan (auto-generate jika None)
            step              : nomor langkah pipeline
            role              : nama agen
            kv_n_selective    : jumlah token untuk selective cache
            latent_steps      : override jumlah latent steps (None = pakai default engine)
        """
        _conv_id = conv_id or str(uuid.uuid4())[:8]
        _max_tok = max_new_tokens or self.max_new_tokens
        _temp    = temperature    or self.temperature
        _top_p   = top_p          or self.top_p
        # [B11 — jalur ini kini HIDUP]. Sebelumnya `json_schema` tak pernah
        #   dioper oleh siapa pun sehingga guided decoding jadi kode mati;
        #   construct hanya memakai `json_mode` (ekstraksi PASCA-generasi, yang
        #   tak bisa menyelamatkan output yang sudah menyimpang). AgentSpec kini
        #   punya field `json_schema` dan prompts.yaml mengisinya untuk construct.
        # ── Build guided-decoding prefix_fn (opsional) ──────────────────────
        # Jika `json_schema` di-supply, bangun prefix_allowed_tokens_fn dari
        # lm-format-enforcer — di setiap step dekoder, hanya token yang
        # melanjutkan parse JSON valid terhadap schema yang boleh keluar.
        # Diterapkan hanya pada mode yang generate text (bukan kv_only).
        _prefix_fn = None
        if json_schema is not None and mode != "kv_only":
            from llm.guided_decoding import build_guided_json_prefix_fn
            _prefix_fn = build_guided_json_prefix_fn(
                self._engine.tokenizer, json_schema,
            )
            print(
                f"[GuidedDecoding] role={role}, mode={mode}: "
                f"enforcing JSON schema via prefix_allowed_tokens_fn"
            )

        # ── Prefill pembuka objek JSON ─────────────────────────────────────
        # Agen yang mewarisi KV berisi objek JSON utuh dari agen hulu cenderung
        # melanjutkan seolah masih berada di dalam objek itu: keluarannya mulai
        # dari NILAI, tanpa `{` pembuka, sehingga `_fix_json` (yang mencari
        # `{` pertama) tak menemukan apa pun dan seluruh jalan tercatat gagal.
        # Terukur pada lengan faktor 2026-08-10: empat dari lima sel
        # `kv_and_text` menghasilkan nol ekspresi karena ini. Bukan pemotongan
        # oleh harness — `text_len` yang dicatat sama persis dengan panjang
        # yang dibangkitkan model.
        #
        # Prefill menutupnya dengan biaya nol: pembuka objek dikirim sebagai
        # bagian giliran asisten, bukan diminta lewat prompt, sehingga model
        # tak punya kesempatan melewatinya. Alternatifnya guided decoding,
        # yang menjamin lebih banyak tetapi menambah 10--20% latensi per token
        # — dan yang selama ini hanya AKTIF bila env `LATENTMAS_GUIDED=1`
        # di-set oleh `pipeline/loop.py`, modul yang sudah dihapus bersama
        # pipeline evolusi. Jadi ia tak pernah menyala pada run mana pun.
        #
        # Isinya datang dari `prefill:` di prompts.yaml, bukan dari sini:
        # teks pembuka yang tepat bergantung pada kontrak keluaran agen, dan
        # lapisan ini tak boleh tahu skema agen mana pun.
        #
        # Diterapkan di SEMUA medium yang membangkitkan teks, termasuk medium
        # yang keluarannya sudah sehat. Menerapkannya hanya pada medium
        # bermasalah akan membuat prosedur pembangkitan berbeda antar-medium,
        # dan perbandingan medium adalah salah satu pertanyaan penelitian.
        # Untuk keluaran yang memang sudah utuh, `_join_prefill` membuangnya
        # lagi, sehingga penerapan seragam ini tak mengubah sel yang sehat.
        _prefill = prefill if mode != "kv_only" else ""

        # Pipeline monitor (safe — no-op if unavailable)
        try:
            from debug import get_monitor as _get_mon
            _mon = _get_mon(auto_create=False)
        except ImportError:
            _mon = None

        # ── Strip VAR markers before engine sees the messages ──────────────
        # `messages_marked` preserves the [[VAR:name]]...[[/VAR:name]] wrappers
        # injected by prompt loaders so `_save_output_snapshot` can annotate
        # which slice came from which YAML variable. The engine always works
        # on the stripped copy — the model never sees markers.
        try:
            from utils.prompt_markers import strip_messages as _strip_msgs
            messages_marked = messages
            messages = _strip_msgs(messages)
        except Exception:
            messages_marked = messages  # fail open: no annotation, no crash

        _run_t0 = time.time()
        _input_tok_count = 0
        try:
            prompt_text = self._engine.format_messages(messages)
            _ids_tmp, _ = self._engine.tokenize(prompt_text)
            _input_tok_count = _ids_tmp.shape[-1] if _ids_tmp is not None else 0
        except Exception:
            pass
        if _mon:
            _mon.track_llm_call_start(
                caller=role, mode=mode,
                input_tokens=_input_tok_count,
                temperature=_temp or 0.0,
                latent_steps=latent_steps or self._engine.latent_steps,
                has_past_kv=past_key_values is not None,
            )

        with self._lock:
            result = LLMResult(mode=mode)
            # [sebagian terjawab — skripsi Bab 4 §Penyaringan KNN]: rumus kosinus
            #   s_i = <K_i, q>/(||K_i|| ||q||) sudah ditulis. TODO(verifikasi): alasan
            #   inkompatibilitas saat latent_steps>0 belum tuntas — hipotesis: re-rotasi
            #   RoPE token terpilih bentrok dengan posisi virtual-token laten. Perlu cek.
            # ── Normalize past_kv ke DynamicCache (boundary masuk) ────
            # Engine internal bekerja penuh dalam DynamicCache; konversi
            # dilakukan SEKALI di sini (bukan di setiap method _CoreEngine)
            # untuk menghindari alokasi wrapper berulang & fragmentasi.
            if past_key_values is not None and isinstance(past_key_values, tuple):
                from transformers import DynamicCache
                past_key_values = DynamicCache(ddp_cache_data=past_key_values)

            # ── KNN filter logging ────────────────────────────────────
            if (self._engine.knn_enabled
                    and past_key_values is not None
                    and _past_length(past_key_values) > 0):
                _pre_len = _past_length(past_key_values)
                _post_len = max(
                    int(_pre_len * self._engine.knn_percentage),
                    self._engine.knn_min_keep,
                )
                _post_len = min(_post_len, _pre_len)
                print(
                    f"[KNN] role={role}, mode={mode}, "
                    f"past_kv: {_pre_len} → ~{_post_len} tokens "
                    f"({self._engine.knn_percentage:.0%}, "
                    f"strategy={self._engine.knn_strategy})"
                )

            # ── kv_only ────────────────────────────────────────────────
            if mode == "kv_only":
                # add_generation_prompt=False — SAMA seperti branch kv_and_text.
                # Tanpa ini KV mewarisi '<|im_start|>assistant\n' menggantung
                # (turn tak pernah ditutup karena kv_only tak generate teks) dan
                # prompt agen berikutnya ter-append DI DALAM turn itu → struktur
                # template rusak bertumpuk tiap hop → repetition collapse pada
                # agen teks pertama (MONITORING_NOTES B10/B11). kv_only kini =
                # kv_and_text minus langkah generate, sesuai desain eksperimen.
                kv, last_hidden, latent_vecs = self._engine.latent_pass(
                    messages, past_key_values, record_vecs=record_latent_vecs,
                    latent_steps=latent_steps,
                    add_generation_prompt=False,
                )
                result.kv_cache    = kv
                result.hidden_last = last_hidden
                result.latent_vecs = latent_vecs
                result.n_latent_steps = self._engine.last_latent_steps_run
                result.latent_stop    = self._engine.last_latent_stop

                prompt = self._engine.format_messages(messages)
                ids, _ = self._engine.tokenize(prompt)
                result.input_ids = ids

            # ── text_only ──────────────────────────────────────────────
            elif mode == "text_only":
                text, ids, out_ids, _ = self._engine.generate_text(
                    messages, past_key_values,
                    max_new_tokens=_max_tok, temperature=_temp, top_p=_top_p,
                    return_kv=False,
                    prefix_allowed_tokens_fn=_prefix_fn,
                    prefill=_prefill,
                )
                if json_mode:
                    text = self._fix_json(text)
                result.text       = text
                result.input_ids  = ids
                result.output_ids = out_ids

            # ── kv_and_text ────────────────────────────────────────────
            elif mode == "kv_and_text":
                # Step 1: Latent pass TANPA assistant prefix.
                #   KV berisi: [prompt tokens + latent virtual tokens]
                #   Model "berpikir diam" di ruang latent sebelum menjawab.
                _t_latent = time.time()
                kv, last_hidden, latent_vecs = self._engine.latent_pass(
                    messages, past_key_values, record_vecs=record_latent_vecs,
                    latent_steps=latent_steps,
                    add_generation_prompt=False,
                )
                result.latent_s = round(time.time() - _t_latent, 3)
                result.n_latent_steps = self._engine.last_latent_steps_run
                result.latent_stop    = self._engine.last_latent_stop

                # Step 2: Generate teks dari KV TANPA re-encode pesan.
                #   Hanya kirim assistant prefix tokens (e.g. <|im_start|>assistant\n)
                #   sebagai trigger. Menghindari double-encoding prompt.
                #
                # Catatan KV-chain (ANTI-PATTERN CONTAMINATION):
                #   HuggingFace DynamicCache dimutasi in-place oleh
                #   model.generate — `kv` hasil latent_pass akan ter-append
                #   assistant prefix + answer tokens setelah generate selesai.
                #   Kalau seluruh KV ini di-chain ke step berikutnya, answer
                #   tokens step sekarang menjadi konteks-terdekat bagi step
                #   berikutnya → pattern-match ke schema jawaban sebelumnya
                #   (misal propose flat JSON mem-bias construct yang
                #   seharusnya nested). Maka kita CROP kembali ke panjang
                #   pre-generation sebelum di-store — konsisten dengan
                #   filosofi Latent-MAS: yang di-pass antar agent adalah
                #   latent reasoning (virtual tokens), bukan discrete
                #   answer tokens.
                # [terjawab — skripsi Bab 4 §Pemangkasan Pasca-Generasi (CROP)]:
                #   C ← C[:ell0], ell0 = panjang pra-generasi (prompt + L virtual token);
                #   buang token jawaban diskret agar tak mengontaminasi agen berikutnya.
                _latent_kv_len = _past_length(kv)
                _t_gen = time.time()
                text, prefix_ids, out_ids, _ = self._engine.generate_from_kv(
                    past_kv=kv, messages=messages,
                    max_new_tokens=_max_tok, temperature=_temp, top_p=_top_p,
                    return_kv=False,
                    prefix_allowed_tokens_fn=_prefix_fn,
                    prefill=_prefill,
                )
                result.gen_s = round(time.time() - _t_gen, 3)
                if crop_after_generate:
                    # Perilaku infra default: buang token jawaban → hanya
                    # [prompt + N latent virtual tokens] yang di-chain (anti-
                    # contamination, filosofi Latent-MAS).
                    try:
                        kv.crop(_latent_kv_len)
                    except AttributeError:
                        pass
                else:
                    # NO-CROP (prod): pertahankan jawaban di KV agar agent berikut
                    # membaca output ASLI (hipotesis/palette), bukan cuma vektor
                    # laten. Tutup turn asisten yang menggantung. (DESIGN.md §2.2)
                    self._engine._close_open_turn(kv)
                if json_mode:
                    text = self._fix_json(text)

                # input_ids untuk logging: tokenize full prompt (murah, no forward pass)
                full_prompt = self._engine.format_messages(messages)
                full_ids, _ = self._engine.tokenize(full_prompt)

                result.text        = text
                result.kv_cache    = kv
                result.input_ids   = full_ids
                result.output_ids  = out_ids
                result.hidden_last = last_hidden
                result.latent_vecs = latent_vecs

            # ── Simpan output LLM ke disk SEGERA (flush per-call) ─────
            # Tulis sebelum proses lain — kalau pipeline crash di tensor
            # logging / KV store, snapshot text tetap persisten.
            if result.text is not None and mode != "kv_only":
                _snap_dur = time.time() - _run_t0
                _snap_out_tok = (
                    result.output_ids.shape[-1]
                    if result.output_ids is not None else 0
                )
                self._save_output_snapshot(
                    conv_id=_conv_id, step=step, role=role, mode=mode,
                    messages=messages_marked, text=result.text,
                    temperature=_temp, has_past_kv=past_key_values is not None,
                    input_tokens=_input_tok_count,
                    output_tokens=_snap_out_tok,
                    duration_s=round(_snap_dur, 4),
                )

            # ── Logging tensor ─────────────────────────────────────────
            if self._conv_mgr is not None:
                self._conv_mgr.save(ConvRecord(
                    conv_id     = _conv_id,
                    step        = step,
                    role        = role,
                    input_ids   = result.input_ids,
                    output_ids  = result.output_ids,
                    hidden_last = result.hidden_last,
                    latent_vecs = result.latent_vecs,
                    metadata    = {
                        "mode" : mode,
                        "model": self._engine.model_name,
                        "ts"   : time.time(),
                    },
                ))
            # [terjawab — skripsi Bab 4 §Pemotongan (truncation)]:
            #   K_l ← K_l[..., -n:, :], V_l ← V_l[..., -n:, :] (pertahankan n token terakhir).
            # ── In-memory KV-cache truncation ─────────────────────────
            if result.kv_cache is not None and self._kv_max_seq_len is not None:
                # `model` dioper agar key di-re-rotasi ke posisi kontigu (B8);
                # tanpa itu setiap pemotongan men-desync RoPE seluruh konteks.
                result.kv_cache = kv_truncate(
                    result.kv_cache, self._kv_max_seq_len, model=self._engine.model
                )

            # ── Simpan KV-cache ke disk ────────────────────────────────
            if self._kv_store is not None and result.kv_cache is not None:
                self._kv_store.save_full(
                    _conv_id, step, result.kv_cache, metadata={"role": role, "mode": mode}
                )
                self._kv_store.save_selective(
                    _conv_id, step, result.kv_cache,
                    n_tokens=kv_n_selective, metadata={"role": role, "mode": mode}
                )
                # Auto-prune disk storage
                self._kv_store.auto_prune(
                    max_total_mb=self._kv_prune_max_mb,
                    max_age_seconds=self._kv_prune_max_age_secs,
                    max_entries_per_tier=self._kv_prune_max_entries,
                )

        # ── Pipeline monitor: record LLM call end + quality ──────────
        if _mon:
            try:
                _run_dur = time.time() - _run_t0
                _out_tok = result.output_ids.shape[-1] if result.output_ids is not None else 0
                _tok_sec = _out_tok / _run_dur if _run_dur > 0 else 0
                _mon.track_llm_call_end(
                    caller=role, duration_s=_run_dur,
                    output_tokens=_out_tok, tokens_per_sec=_tok_sec,
                    total_tokens=_input_tok_count + _out_tok,
                    mode=mode,
                )
                if result.text:
                    _out_ids_list = result.output_ids[0].tolist() if result.output_ids is not None else None
                    _mon.analyze_llm_output(result.text, caller=role, token_ids=_out_ids_list)
            except Exception:
                pass

        return result

    # ── Shortcut methods ─────────────��────────────────────────────────────

    def run_latent_pass(
        self,
        messages       : List[Dict[str, str]],
        past_key_values: Optional[KVCache] = None,
        **kwargs
    ) -> LLMResult:
        """Shortcut mode='kv_only'. Untuk agen yang hanya membangun konteks."""
        return self.run(messages, mode="kv_only",
                        past_key_values=past_key_values, **kwargs)

    def run_generate(
        self,
        messages       : List[Dict[str, str]],
        past_key_values: Optional[KVCache] = None,
        return_kv      : bool = False,
        **kwargs
    ) -> LLMResult:
        """
        Shortcut untuk generate teks.
        return_kv=False -> text_only
        return_kv=True  -> kv_and_text
        """
        mode = "kv_and_text" if return_kv else "text_only"
        return self.run(messages, mode=mode,
                        past_key_values=past_key_values, **kwargs)

    # ── Session (kompatibel dengan ChatSession di client.py) ──────────────

    def build_chat_session(
        self,
        conversation_id      : Optional[str] = None,
        session_system_prompt: Optional[str] = None,
    ) -> "LocalChatSession":
        """Buat session multi-turn. Compatible dengan APIBackend.build_chat_session."""
        return LocalChatSession(
            backend=self,
            conversation_id=conversation_id,
            system_prompt=session_system_prompt,
        )

    # ── Utilitas ──────────────────────────────────────────────────────────

    def load_kv(self, conv_id: str, step: int,
                tier: str = "full") -> Optional[KVCache]:
        """Load KV-cache dari disk untuk resume evolution."""
        if self._kv_store is None:
            raise RuntimeError("store_kv=False, KVCacheStore tidak aktif.")
        return self._kv_store.load(conv_id, step, tier,
                                   device=self._engine.device)

    def debug_step(self, conv_id: str, step: int, role: str) -> Dict[str, str]:
        """Decode tensor log satu langkah untuk debugging."""
        if self._conv_mgr is None:
            return {"error": "log_tensors=False, TensorConvManager tidak aktif."}
        return self._conv_mgr.decode_step(
            conv_id, step, role, self._engine.tokenizer
        )

    def kv_store_info(self, conv_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List metadata KV-cache yang tersimpan."""
        if self._kv_store is None:
            return []
        return self._kv_store.list_entries(conv_id)

    def kv_total_size_mb(self) -> float:
        if self._kv_store is None:
            return 0.0
        return self._kv_store.total_size_mb()

    def get_info(self) -> Dict[str, Any]:
        """Info singkat untuk monitoring."""
        info: Dict[str, Any] = {
            "model"        : self._engine.model_name,
            "device"       : str(self._engine.device),
            "latent_steps" : self._engine.latent_steps,
            "use_realign"  : (self._engine.realigner is not None and
                              self._engine.realigner.use_realign),
            "thinking"     : self._engine.enable_thinking,
            "log_tensors"  : self._conv_mgr is not None,
            "store_kv"     : self._kv_store is not None,
        }
        if self._kv_store:
            info["kv_store_mb"] = round(self.kv_total_size_mb(), 2)
        return info

    # ── Kompatibilitas APIBackend: token counting & embedding ──────────

    def build_messages_and_calculate_token(
        self,
        user_prompt  : str,
        system_prompt: Optional[str] = None,
    ) -> int:
        """Hitung jumlah token. Compatible dengan APIBackend."""
        msgs = self.build_messages(user_prompt, system_prompt)
        prompt = self._engine.format_messages(msgs)
        ids, _ = self._engine.tokenize(prompt)
        return ids.shape[-1]

    @torch.no_grad()
    def create_embedding(
        self,
        input_content: Union[str, List[str]],
    ) -> List[List[float]]:
        """
        Buat embedding menggunakan model lokal (mean-pool input embeddings).
        Compatible dengan APIBackend.create_embedding.
        """
        if isinstance(input_content, str):
            input_content = [input_content]
        embeddings = []
        embed_layer = self._engine.model.get_input_embeddings()
        for text in input_content:
            encoded = self._engine.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512,
            )
            ids = encoded["input_ids"].to(self._engine.device)
            mask = encoded["attention_mask"].to(self._engine.device)
            with self._lock:
                emb = embed_layer(ids)  # [1, seq, dim]
            # mean pool over non-padding tokens
            mask_expanded = mask.unsqueeze(-1).float()
            pooled = (emb * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-6)
            embeddings.append(pooled.squeeze(0).float().cpu().tolist())
        return embeddings

    def _save_output_snapshot(
        self,
        conv_id        : str,
        step           : int,
        role           : str,
        mode           : str,
        messages       : List[Dict[str, str]],
        text           : str,
        temperature    : float,
        has_past_kv    : bool,
        input_tokens   : Optional[int] = None,
        output_tokens  : Optional[int] = None,
        duration_s     : Optional[float] = None,
    ) -> None:
        """Tulis 1 file .md per call_llm + append 1 baris ke index.jsonl.

        Format Markdown human-readable:
          - Frontmatter-style metadata block
          - "## Variables" section listing each [[VAR:name]]...[[/VAR:name]]
            slice yang ditemukan di prompt (sumber dari YAML placeholder)
          - "## System Prompt" dan "## User Prompt" — isi prompt dengan
            marker yang di-render jadi `<<<name>>>...<<</name>>>` agar user
            bisa langsung lihat slice mana datang dari variabel mana
          - "## Response" — text mentah dari LLM

        Markers HANYA tampil di file ini; LLM menerima prompt yang sudah
        di-strip oleh `run()` sebelum engine call.

        Ditulis dengan flush+fsync supaya crash di tengah iterasi tetap
        meninggalkan jejak. Exception di-swallow agar logging tidak pernah
        mematikan pipeline.
        """
        if self._output_log_dir is None:
            return
        try:
            from utils.prompt_markers import render_for_debug
        except Exception:
            render_for_debug = None  # fail open

        try:
            sys_prompt = ""
            usr_prompt = ""
            for m in messages:
                if m.get("role") == "system" and not sys_prompt:
                    sys_prompt = m.get("content", "") or ""
                elif m.get("role") == "user":
                    usr_prompt = m.get("content", "") or ""

            # Counter global (thread-safe) — semua instance/modul berbagi satu
            # urutan lewat atribut modul `engine` (lihat catatan header modul
            # ini soal kenapa `global` polos tak cukup lintas-modul).
            with engine._GLOBAL_OUTPUT_LOCK:
                engine._GLOBAL_CALL_COUNTER += 1
                n = engine._GLOBAL_CALL_COUNTER

            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            # Nama file: counter saja — run-ts sudah ada di session folder
            filename = f"{n:04d}_{role}_{mode}.md"
            filepath = self._output_log_dir / filename

            # Render markers → human-readable inline labels
            if render_for_debug is not None:
                sys_annotated, sys_vars = render_for_debug(sys_prompt)
                usr_annotated, usr_vars = render_for_debug(usr_prompt)
            else:
                sys_annotated, sys_vars = sys_prompt, []
                usr_annotated, usr_vars = usr_prompt, []

            # Build "Variables" section: deduplicate by name (keep first value)
            all_vars: List[Tuple[str, str]] = []
            seen_names: set = set()
            for name, value in (sys_vars + usr_vars):
                if name in seen_names:
                    continue
                seen_names.add(name)
                all_vars.append((name, value))

            md_parts: List[str] = []
            md_parts.append(f"# Call {n:04d} — `{role}` ({mode})\n")
            md_parts.append("## Meta\n")
            md_parts.append(f"- ts: {ts}")
            md_parts.append(f"- conv_id: `{conv_id}`")
            md_parts.append(f"- step: {step}")
            md_parts.append(f"- temperature: {temperature}")
            md_parts.append(f"- has_past_kv: {has_past_kv}")
            md_parts.append(f"- input_tokens: {input_tokens}")
            md_parts.append(f"- output_tokens: {output_tokens}")
            md_parts.append(f"- duration_s: {duration_s}")
            md_parts.append(f"- text_len: {len(text)}\n")

            if all_vars:
                md_parts.append("## Variables (dari YAML placeholder)\n")
                for name, value in all_vars:
                    preview = value if len(value) <= 200 else value[:200] + "…"
                    preview = preview.replace("\n", " ⏎ ")
                    md_parts.append(f"- **{name}** ({len(value)} chars): {preview}")
                md_parts.append("")

            md_parts.append("## System Prompt\n")
            md_parts.append("```text")
            md_parts.append(sys_annotated)
            md_parts.append("```\n")

            md_parts.append("## User Prompt\n")
            md_parts.append("```text")
            md_parts.append(usr_annotated)
            md_parts.append("```\n")

            md_parts.append("## Response\n")
            md_parts.append("```text")
            md_parts.append(text)
            md_parts.append("```")

            content = "\n".join(md_parts) + "\n"
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())

            # Index ringan: 1 baris JSON per call, untuk scan tanpa baca semua file
            if engine._GLOBAL_OUTPUT_INDEX is not None:
                index_entry = {
                    "n"            : n,
                    "ts"           : ts,
                    "role"         : role,
                    "mode"         : mode,
                    "step"         : step,
                    "conv_id"      : conv_id,
                    "text_len"     : len(text),
                    "duration_s"   : duration_s,
                    "file"         : filename,
                    "vars"         : [name for name, _ in all_vars],
                }
                with engine._GLOBAL_OUTPUT_LOCK:
                    with open(engine._GLOBAL_OUTPUT_INDEX, "a", encoding="utf-8") as f:
                        f.write(json.dumps(index_entry, ensure_ascii=False) + "\n")
                        f.flush()
                        os.fsync(f.fileno())
        except Exception:
            # Logging TIDAK boleh mematikan pipeline
            pass

    @staticmethod
    def _fix_json(text: str) -> str:
        """Ekstrak + perbaiki JSON dari output LLM.

        Strategi (urut prioritas):
            1. Parse full text apa adanya — output guided-decoding biasanya
               sudah valid utuh, tidak butuh ekstraksi.
            2. Strip code fence ```json ... ``` jika ada.
            3. Outermost match: first `{` → last `}`. Ini menangani kasus
               "prose + JSON di akhir" tanpa mengorbankan nested structure.
            4. Balanced-bracket scan dari kanan: untuk setiap `}` paling
               kanan, hitung mundur sampai kurung balanced — itulah `{`
               struktural pasangannya. Mengganti `rfind` greedy yang dulu
               salah (selalu landing di inner-dict `variables`).
        """
        def _try_parse(s: str) -> Optional[str]:
            try:
                json.loads(s)
                return s
            except json.JSONDecodeError:
                return _try_latex_fix(s)

        def _try_latex_fix(s: str) -> Optional[str]:
            fixed = s
            for cmd in ["text","frac","left","right","times","cdot",
                        "sqrt","sum","prod","int","alpha","beta","gamma","delta"]:
                fixed = re.sub(rf"(?<!\\)\\({cmd})", r"\\\\\1", fixed)
            fixed = re.sub(r"(?<!\\)\\([_\{}\[\]])", r"\\\\\1", fixed)
            try:
                json.loads(fixed)
                return fixed
            except json.JSONDecodeError:
                return None

        t = text.strip()

        # 1. Try the whole thing — guided decoding sudah memproduksi JSON valid.
        if (parsed := _try_parse(t)) is not None:
            return parsed

        # 2. Strip ```json fence``` jika ada.
        fence = re.search(r"```json\s*(.*?)```", t, re.DOTALL | re.IGNORECASE)
        if fence:
            t = fence.group(1).strip()
            if (parsed := _try_parse(t)) is not None:
                return parsed

        # 3. Outermost {…} match: first `{` → last `}`.
        first_open = t.find("{")
        last_close = t.rfind("}")
        if first_open != -1 and last_close > first_open:
            if (parsed := _try_parse(t[first_open:last_close + 1])) is not None:
                return parsed

        # 4. Balanced-bracket scan dari kanan. Untuk tiap `}` paling kanan,
        # cari `{` yang BALANCE secara struktural (bukan rfind greedy yang
        # selalu landing di inner-dict).
        end = len(t)
        while end > 0:
            e = t.rfind("}", 0, end)
            if e == -1:
                break
            depth = 0
            s = -1
            for i in range(e, -1, -1):
                ch = t[i]
                if ch == "}":
                    depth += 1
                elif ch == "{":
                    depth -= 1
                    if depth == 0:
                        s = i
                        break
            if s == -1:
                break
            if (parsed := _try_parse(t[s:e + 1])) is not None:
                return parsed
            end = e

        return text


# =============================================================================
# FACTORY
# =============================================================================

def get_local_backend(
    model_name  : str  = "Qwen/Qwen3-4B",
    latent_steps: int  = 0,
    use_realign : bool = False,
    device      : str  = "cuda",
    **kwargs,
) -> LocalLLMBackend:
    """
    Factory untuk LocalLLMBackend.

    Contoh:
        # Mode standar (drop-in APIBackend, tanpa latent)
        backend = get_local_backend()

        # Mode latent 5 steps dengan realignment
        backend = get_local_backend(latent_steps=5, use_realign=True)

        # Mode latent + simpan KV ke disk (untuk evolution loop)
        backend = get_local_backend(latent_steps=5, store_kv=True, kv_dir="./kv")
    """
    return LocalLLMBackend(
        model_name=model_name, device=device,
        latent_steps=latent_steps, use_realign=use_realign,
        **kwargs,
    )

