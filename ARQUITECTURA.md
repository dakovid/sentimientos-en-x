## Arquitectura de la Solución — Análisis de Sentimiento ES (X/Twitter) sin API

### Resumen ejecutivo
Solución mínima y ejecutable que ingesta posts públicos de X sin usar la API oficial (vía `snscrape` por CLI o CSV), calcula sentimiento en español con enfoque léxico + reglas simples, y produce KPIs, top de términos y dos gráficos. Se prioriza simplicidad, claridad y robustez en un único archivo `app.py`.

### Alcance y restricciones
- **Sin API oficial**: ingesta por `snscrape` (CLI) o `CSV`.
- **Self-contained**: un solo archivo `app.py` (Python 3.10+), dependencias mínimas (`pandas`, `numpy`, `matplotlib`).
- **Gráficos**: `matplotlib` (sin seaborn).
- **Outputs**: CSV/PNG en carpeta `salida`.
- **Documentación**: diagramas Mermaid y PlantUML incluidos.

### Arquitectura lógica (alto nivel)
- Ver diagrama Mermaid: `diagrams/arquitectura-mermaid.md`.
- Ver diagrama de secuencia (PlantUML): `diagrams/flujo-secuencia.puml`.

Flujo principal: CLI → Ingesta (sample/sn/csv) → Normalización → Léxico (frases, luego palabras) → Score normalizado → Etiqueta → KPIs/Top términos → CSV/PNG.

### Decisiones de diseño y justificación
- **Ingesta por `snscrape` vía `subprocess`**: evita la API oficial y dependencias externas complejas; si falta la herramienta, el mensaje guía al usuario a `pip install snscrape`. Alternativamente, `CSV` garantiza operatividad offline y reproducibilidad. Un **dataset `sample`** asegura experiencia out-of-the-box.
- **Un único archivo `app.py`**: facilita bootstrap, revisión y ejecución inmediata. Se compensa la falta de módulos con funciones pequeñas, nombres claros y docstrings, manteniendo SRP.
- **NLP léxico + reglas**: 
  - Procesa **frases primero** (peso 1.5) y **palabras después** (peso 1.0) para capturar negaciones y expresiones multi-palabra (p.ej., “no funciona”, “vale la pena”).
  - **Normalización por longitud** con \( score/\sqrt{tokens + usados} \) evita sesgo por textos largos.
  - Ventajas: transparente, rápido, reproducible y sin dependencias pesadas. Trade-off: menor cobertura semántica que modelos ML.
- **Normalización de texto**: minúsculas, sin acentos y espacios compactados. Reduce variabilidad y mejora matching de léxico.
- **KPIs y top de términos**: métricas simples y útiles para una primera lectura; `Counter` + stopwords básicas permiten identificar señales sin añadir complejidad.
- **Gráficos con `matplotlib`**: cumple con la restricción de dependencias y genera rápidamente barras (distribución de sentimiento) e histograma (scores).
- **Persistencia en CSV/PNG**: formatos abiertos, simples de integrar y portables. Evitamos bases de datos para mantener la huella mínima.
- **CLI con `argparse`**: interfaz clara, validación de modos y mensajes de error explícitos.

### Calidad, mantenibilidad y seguridad
- **Calidad**: PEP8, type hints, docstrings, funciones cohesionadas y sin efectos colaterales innecesarios.
- **Mantenibilidad**: separación por funciones (ingesta, NLP, KPIs, gráficos, guardado). Léxico centralizado y fácil de extender.
- **Seguridad**: sin credenciales ni secretos; ejecución local. Se limpia texto (URLs, menciones/hashtags) para el conteo de términos. No se exponen datos sensibles.

### Rendimiento y escalabilidad
- **Rendimiento**: O(n) sobre cantidad de posts; `pandas` optimiza operaciones tabulares. Conteo de términos con `Counter` eficiente.
- **Escalabilidad**: para mayores volúmenes, se puede:
  - Procesar por lotes/streaming (chunks) en `pandas`.
  - Exportar particionado por fecha/tema.
  - Containerizar y escalar horizontalmente (múltiples instancias por partición de consulta/fecha).

### Operación y observabilidad
- **Operación**: comandos simples; si `snscrape` no está, se informa cómo instalar. Errores con mensajes claros para CSV faltante o sin resultados.
- **Observabilidad**: logs a consola suficientes para un primer corte; se puede extender a logging estructurado en el futuro.

### Trade-offs relevantes
- **Léxico/reglas vs. modelos ML**: se prioriza transparencia, cero dependencias pesadas y ejecución offline. A costa de menor cobertura semántica, es ideal para un MVP o entornos con restricciones.
- **`subprocess` vs. SDK/API**: `subprocess` con `snscrape` elimina necesidad de credenciales y cumple la restricción de “sin API oficial”. Depende del binario en PATH; mitigado con mensaje de instalación.
- **Un solo archivo vs. módulos**: simplifica entrega inicial; para crecer, se puede extraer módulos (ingesta, nlp, salida) manteniendo contratos estables.

### Futuras mejoras
- Ampliar léxico (sinónimos, intensificadores/atenuadores, emojis), detección de negación más sofisticada.
- Lematización/stemming ligero (siempre que no introduzca dependencias pesadas).
- Modelos livianos offline (cuando la restricción lo permita) y evaluación A/B sobre KPIs.
- Exportación en Parquet, particionado temporal, y reporting adicional (series temporales por día/tema).
- Dockerización opcional con `python:3.10-slim` y volumen para `salida/` para facilitar portabilidad.

### Ejecución (recordatorio)
- `python app.py`
- `python app.py --modo sn --query "DevOps Argentina" --desde 2025-08-01 --limite 300`
- `python app.py --modo csv --csv_in data/posts.csv`
