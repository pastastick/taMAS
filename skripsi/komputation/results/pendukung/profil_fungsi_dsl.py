"""Profil waktu tiap fungsi DSL pada data pasar NYATA (lampiran skripsi).

Dipakai untuk memutuskan fungsi mana yang dibuang dari DSL. Tiap fungsi diberi
BATAS WAKTU: tanpa itu, fungsi pathologis (`rolling().apply` dengan callback
Python) menahan profil berjam-jam dan tabelnya tak pernah jadi. Yang melewati
batas dicatat sebagai ">CAP dtk" — untuk keputusan buang/pakai, "lebih lambat
dari anggaran skoring" sudah cukup informatif; angka pastinya tidak.

Angka di sini diukur pada korpus PENUH yang dipakai `eval/ic.py` (~4.370 saham
x 243 hari OOS), yaitu beban yang sama dengan skoring produksi — bukan sampel
kecil yang akan memberi kesan terlalu optimistis.

    PYTHONPATH=backend python results/pendukung/profil_fungsi_dsl.py
"""
import json
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
from eval.ic import Lab  # noqa: E402

CAP = 120  # detik; anggaran skoring produksi 90 dtk (run_factor.score_expressions)

UJI = {
    'RANK': 'RANK($volume)',
    'TS_MEAN': 'TS_MEAN($close,60)',
    'TS_STD': 'TS_STD($close,60)',
    'TS_ZSCORE': 'TS_ZSCORE($volume,60)',
    'TS_RANK': 'TS_RANK($close,60)',
    'TS_PCTCHANGE': 'TS_PCTCHANGE($volume,60)',
    'TS_MAX': 'TS_MAX($close,60)',
    'TS_MIN': 'TS_MIN($close,60)',
    'TS_SUM': 'TS_SUM($volume,60)',
    'TS_ARGMAX': 'TS_ARGMAX($close,60)',
    'TS_ARGMIN': 'TS_ARGMIN($close,60)',
    'TS_MEDIAN': 'TS_MEDIAN($close,60)',
    'TS_VAR': 'TS_VAR($close,60)',
    'DELAY': 'DELAY($close,60)',
    'DELTA': 'DELTA($close,60)',
    'SIGN': 'SIGN($return)',
    'ABS': 'ABS($return)',
    'LOG': 'LOG($volume)',
    'TS_CORR': 'TS_CORR($close,$volume,60)',
    'TS_COVARIANCE': 'TS_COVARIANCE($close,$volume,60)',
    'TS_QUANTILE': 'TS_QUANTILE($volume,60,0.9)',
    'DECAYLINEAR': 'DECAYLINEAR($close,60)',
    'WMA': 'WMA($close,60)',
    'HIGHDAY': 'HIGHDAY($close,60)',
    'LOWDAY': 'LOWDAY($close,60)',
    # Tersangka pathologis (semuanya muncul di ekspresi yang kena timeout>90s
    # pada run 2026-08-10 — lihat ragam_eval_error.json).
    'TS_MAD': 'TS_MAD($return,60)',
    'TS_SKEW': 'TS_SKEW($return,60)',
    'TS_KURT': 'TS_KURT($return,60)',
    'REGRESI': 'REGRESI($return,$volume,60)',
    'REGBETA': 'REGBETA($return,$volume,60)',
    'SEQUENCE': 'TS_MEAN(SEQUENCE(60),5)',
}


class _Timeout(Exception):
    pass


def _alarm(signum, frame):  # noqa: ARG001
    raise _Timeout()


def main() -> None:
    lab = Lab(mode='fast')
    _ = lab.df  # muat data sekali, di luar pengukuran
    hasil = {}
    signal.signal(signal.SIGALRM, _alarm)
    for nama, e in UJI.items():
        t = time.time()
        try:
            signal.alarm(CAP)
            lab.values(e)
            signal.alarm(0)
            dt, status = time.time() - t, 'ok'
        except _Timeout:
            signal.alarm(0)
            dt, status = float(CAP), f'>{CAP}s'
        except Exception as ex:  # noqa: BLE001
            signal.alarm(0)
            dt, status = time.time() - t, type(ex).__name__
        hasil[nama] = {'detik': round(dt, 2), 'status': status, 'ekspresi': e}
        print(f'{nama:16s} {dt:8.2f} dtk  {status}', flush=True)

    p = Path(__file__).parent
    (p / 'profil_fungsi_dsl.json').write_text(
        json.dumps(hasil, indent=2, ensure_ascii=False))
    L = ['# Profil waktu fungsi DSL',
         '',
         'Korpus penuh `eval/ic.py` (~4.370 saham x 243 hari OOS) — beban yang',
         f'sama dengan skoring produksi. Batas {CAP} dtk per fungsi; anggaran',
         'skoring produksi 90 dtk, jadi apa pun yang mendekati/melewatinya akan',
         'hilang sebagai `timeout>90s` tergantung beban mesin.',
         '',
         '| fungsi | detik | status |', '|---|---:|---|']
    for k, v in sorted(hasil.items(), key=lambda x: -x[1]['detik']):
        L.append(f"| {k} | {v['detik']} | {v['status']} |")
    (p / 'profil_fungsi_dsl.md').write_text('\n'.join(L) + '\n')

    print('\n=== TERLAMBAT ===')
    for k, v in sorted(hasil.items(), key=lambda x: -x[1]['detik'])[:8]:
        print(f"  {v['detik']:8.2f} dtk  {k}  {v['status']}")


if __name__ == '__main__':
    main()
