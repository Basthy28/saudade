from pathlib import Path

import pandas as pd
import numpy as np
import os

ROOT = Path(os.environ.get("SAUDADE_ROOT", Path(__file__).resolve().parents[2]))
DATA = Path(os.environ.get("SAUDADE_DATA", ROOT / "data"))

RANK_DIR = DATA / "01_extraction"
(RANK_DIR / "newdata").mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RANK_DIR / "analysis_output" / "origin_labels.csv")
df = df[df['origin'] != 'OUTRO'].copy()
df['backlinks'] = df['backlinks'].fillna(0)

# SCENARIO G: Culturally aware penalization + Strict quality floor
# BR=0.15 (Novelas da Globo passam, sub-celebridades não)
# PALOP=0.20 (Machado de Assis / Autores entram)
# INT=0.10 (Jesus Cristo e Beatles...)
fatores = {'PT':1.0, 'PALOP':0.20, 'BR':0.15, 'INT':0.10}

# 1. Aplicar Hard Floor PRIMEIRO
# Teto mínimo de 15 backlinks para todos.
df_filtered = df[df['backlinks'] >= 15].copy()
removidos = len(df) - len(df_filtered)

# 2. Calcular Score
df_filtered['score'] = df_filtered.apply(lambda r: np.log1p(r['backlinks']) * fatores.get(r['origin'], 1.0), axis=1)
df_filtered = df_filtered.sort_values(['category', 'score'], ascending=[True, False]).reset_index(drop=True)

# 3. Retenção (50%)
kept = []
for cat, grp in df_filtered.groupby('category'):
    n = len(grp)
    cutoff_idx = max(int(n * 0.50), 5)  
    slice_50 = grp.iloc[:cutoff_idx]
    kept.append(slice_50)
    
final_df = pd.concat(kept)

output_path = RANK_DIR / "newdata" / "final_ranking.csv"
final_df.to_csv(output_path, index=False)

pt = sum(final_df['origin'] == 'PT')
int_ = sum(final_df['origin'] == 'INT')
br = sum(final_df['origin'] == 'BR')
palop = sum(final_df['origin'] == 'PALOP')

print(f"Lixo (<15 BL) removido inicialmente: {removidos} entidades.")
print(f"TOTAL FINAL guardado em '{output_path}': {len(final_df)} entidades")
print("=== Distribuição ===")
print(f"PT:    {pt:4d} ({(pt/len(final_df))*100:.1f}%)")
print(f"INT:   {int_:4d} ({(int_/len(final_df))*100:.1f}%)")
print(f"BR:    {br:4d} ({(br/len(final_df))*100:.1f}%)")
print(f"PALOP: {palop:4d} ({(palop/len(final_df))*100:.1f}%)")
