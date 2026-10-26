import os
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("SAUDADE_ROOT", Path(__file__).resolve().parents[2]))
DATA = Path(os.environ.get("SAUDADE_DATA", ROOT / "data"))

def generate_report():
    input_file = DATA / "01_extraction" / "newdata" / "final_ranking.csv"
    output_file = DATA / "01_extraction" / "newdata" / "quality_review.md"
    
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        with open(output_file, 'w') as f:
            f.write(f"# Erro ao ler o ficheiro\n{e}")
        return

    total = len(df)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# 📊 Relatório de Qualidade - Entidades Distiladas\n\n")
        f.write(f"**Total de entidades:** {total}\n\n")
        
        f.write("## 🌍 Distribuição por Origem\n")
        origens = df['origin'].value_counts()
        for orig, count in origens.items():
            f.write(f"- **{orig}**: {count} ({(count/total)*100:.1f}%)\n")
        f.write("\n")
        
        f.write("## 🏷️ Top 5 Categorias (por volume)\n")
        cats = df['category'].value_counts().head(5)
        for cat, count in cats.items():
            f.write(f"- **{cat}**: {count} entidades\n")
        f.write("\n")
            
        f.write("## 🔍 Amostra Aleatória por Origem (Top 3 de cada)\n")
        f.write("Aqui tens alguns exemplos reais que entraram no dataset final, para avaliares se a qualidade e os filtros (ex: novela vs literatura) estão a funcionar como planeado.\n\n")
        
        for orig in origens.index:
            f.write(f"### Origem: {orig}\n")
            sample = df[df['origin'] == orig].sample(min(3, len(df[df['origin'] == orig])), random_state=42)
            for _, row in sample.iterrows():
                f.write(f"- **{row['title']}** (_{row.get('category', 'N/A')}_) | Backlinks: {row.get('backlinks', 0)} | Score: {row.get('score', 0):.2f}\n")
            f.write("\n")
            
        f.write("## 🏆 Top 10 Entidades Absolutas (Maior Score)\n")
        top10 = df.nlargest(10, 'score')
        for i, row in top10.iterrows():
             f.write(f"{i+1}. **{row['title']}** ({row['origin']}) - {row.get('category', 'N/A')} - Score: {row['score']:.2f}\n")

if __name__ == '__main__':
    generate_report()
