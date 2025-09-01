# Arquitectura (simple, sin API oficial)

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
