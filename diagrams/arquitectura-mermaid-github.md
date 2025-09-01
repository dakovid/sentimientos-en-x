# Arquitectura (variante compatible con GitHub)

```mermaid
flowchart LR
  U[Usuario CLI] -->|flags --modo --query --desde| APP[app.py]

  subgraph INGESTA
    SN[snscrape CLI]
    CSV[CSV entrada]
    SMP[Dataset ejemplo]
  end

  APP -->|modo sn| SN
  APP -->|modo csv| CSV
  APP -->|modo sample| SMP

  subgraph PROCESAMIENTO
    NORM[normalizar]
    LEX[score_sentimiento y lexico ES]
    LAB[etiqueta]
    TOP[top_terminos]
    KPI[kpis]
  end

  SN --> APP
  CSV --> APP
  SMP --> APP

  APP --> NORM --> LEX --> LAB
  APP --> TOP
  APP --> KPI

  subgraph SALIDAS
    CSVOUT[posts_con_sentimiento.csv]
    TOPCSV[top_terminos.csv]
    BARIMG[sentimientos_barras.png]
    HISTIMG[scores_hist.png]
  end

  KPI --> CSVOUT
  LAB --> CSVOUT
  TOP --> TOPCSV
  APP --> BARIMG
  APP --> HISTIMG
