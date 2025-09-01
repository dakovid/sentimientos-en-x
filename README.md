## Análisis de Sentimiento ES (X/Twitter) sin API oficial

Script en un solo archivo (`app.py`) para ingestar posts públicos (sample, snscrape o CSV), calcular sentimiento en español (léxico + frases y reglas simples), generar KPIs, top de términos y dos gráficos, y exportar resultados a CSV/PNG.

### Requisitos
```bash
pip install pandas numpy matplotlib
# opcional (solo para --modo sn):
pip install snscrape
```

### Uso rápido
```bash
python app.py
python app.py --modo sn --query "DevOps Argentina" --desde 2025-08-01 --limite 300
python app.py --modo csv --csv_in data/posts.csv
```

### Salidas
- `salida/posts_con_sentimiento.csv`  
- `salida/top_terminos.csv`  
- `salida/sentimientos_barras.png`  
- `salida/scores_hist.png`

### Diagrama (Mermaid)
A continuación se incrusta el diagrama de `diagrams/arquitectura-mermaid.md`:

```mermaid
flowchart LR
  U[Usuario/CLI] -->|flags --modo/--query/--desde| APP[app.py]

  subgraph INGESTA
    SN[snscrape CLI]:::ext
    CSV[(CSV de entrada)]
    SMP[(Dataset de ejemplo)]
  end

  APP -->|--modo sn| SN
  APP -->|--modo csv| CSV
  APP -->|--modo sample| SMP

  subgraph PROCESAMIENTO
    NORM[normalizar(texto)]
    LEX[score_sentimiento()<br/>léxico ES + frases]
    LAB[etiqueta(score)]
    TOP[top_terminos()]
    KPI[kpis()]
  end

  SN --> APP
  CSV --> APP
  SMP --> APP

  APP --> NORM --> LEX --> LAB
  APP --> TOP
  APP --> KPI

  subgraph SALIDAS
    CSVOUT[(posts_con_sentimiento.csv)]
    TOPCSV[(top_terminos.csv)]
    BARIMG[(sentimientos_barras.png)]
    HISTIMG[(scores_hist.png)]
  end

  KPI --> CSVOUT
  LAB --> CSVOUT
  TOP --> TOPCSV
  APP --> BARIMG
  APP --> HISTIMG

  classDef ext fill:#fff,stroke:#888,stroke-width:1px,stroke-dasharray: 3 3;
```

### Diagrama (PlantUML)
Archivo: `diagrams/flujo-secuencia.puml`. Para render localmente, usar una extensión de VS Code o un servidor PlantUML.

### Notas
- No se usa la API oficial de X/Twitter.
- `snscrape` debe estar en el PATH si se usa `--modo sn`.
