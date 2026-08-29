# Geometri vs kinerja hilir (deskriptif, n=5 mode)

Regenerasi: `python scripts/analisis_geometri_kinerja.py`

| mode | cos ke embedding terdekat |
|---|---:|
| raw | 0.3120 |
| soft | 0.9269 |
| gumbel | 0.9848 |
| sample | 1.0000 |
| moi | 1.0000 |

## Korelasi dengan recall kanal laten murni (m=10, k=5, trials=20)

- **dsl**: Spearman ρ=0.975, Pearson r=0.998 (n=5 titik: raw, soft, gumbel, sample, moi)
- **token**: Spearman ρ=0.684, Pearson r=0.820 (n=5 titik: raw, soft, gumbel, sample, moi)

## Korelasi dengan akurasi bench (medium kv, limit=100)

- **arc_challenge**: Spearman ρ=0.433, Pearson r=0.685
- **gsm8k**: Spearman ρ=0.564, Pearson r=0.772
- **humanevalplus**: Spearman ρ=0.975, Pearson r=0.998

## Pembacaan

Korelasi tinggi pada `token`/`humanevalplus` (payload/tugas yang paling menuntut presisi simbolik) dan lemah/tak bermakna pada tugas penalaran umum akan mengkonfirmasi disosiasi §1 README secara numerik, bukan hanya lewat rentang p-value McNemar per tugas. n=5 tetap kecil — baca sebagai ARAH hubungan, bukan bukti bentuknya (itu tugas sumbu `mix`).
