"""
Aplicación simple de análisis de sentimiento en español para posts de X/Twitter sin usar la API oficial.

Uso rápido (Python 3.10+):
    python app.py
    python app.py --modo sn --query "DevOps Argentina" --desde 2025-08-01 --limite 300
    python app.py --modo csv --csv_in data/posts.csv

Dependencias mínimas:
    pip install pandas numpy matplotlib
    # opcional (solo para --modo sn):
    pip install snscrape

Este script ingesta datos desde:
  - sample: dataset de ejemplo (~30 filas)
  - sn: via snscrape (CLI) usando subprocess (sin API oficial)
  - csv: archivo CSV con columnas mínimas (fecha, autor, texto). "tema" es opcional.

Genera:
  - salida/posts_con_sentimiento.csv
  - salida/top_terminos.csv
  - salida/sentimientos_barras.png
  - salida/scores_hist.png

Nota: si usa --modo sn y no posee snscrape, verá el mensaje de instalación sugerido.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import unicodedata


# ==========================
# Utilidades de normalización
# ==========================
def normalizar(texto: str) -> str:
    """Normaliza texto: minúsculas, sin acentos, espacios compactados.

    Args:
        texto: Texto de entrada.
    Returns:
        Texto normalizado en minúsculas, sin acentos y con espacios simples.
    """
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    # Remover acentos
    texto = unicodedata.normalize("NFKD", texto)
    texto = texto.encode("ascii", "ignore").decode("ascii")
    # Compactar espacios
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


# ==========================
# Léxico básico ES
# ==========================
def construir_lexico() -> Tuple[Dict[str, float], Dict[str, float]]:
    """Construye léxico de frases y palabras con pesos.

    Returns:
        Tupla (lexico_frases, lexico_palabras) con pesos positivos/negativos.
    """
    # Frases con orientación explícita
    frases_pos = [
        "vale la pena",
        "anda rapido",
        "excelente soporte",
        "super recomendado",
        "muy bueno",
        "me encanta",
    ]
    frases_neg = [
        "no funciona",
        "no me gusta",
        "no vale la pena",
        "anda lento",
        "mal servicio",
        "atencion pesima",
        "no lo recomiendo",
        "muy malo",
    ]

    palabras_pos = [
        "excelente",
        "genial",
        "top",
        "bueno",
        "rapido",
        "barato",
        "feliz",
        "amor",
        "fix",
        "crack",
    ]
    palabras_neg = [
        "horrible",
        "terrible",
        "malo",
        "lento",
        "caro",
        "triste",
        "odio",
        "bug",
        "estafa",
    ]

    lexico_frases: Dict[str, float] = {normalizar(f): 1.5 for f in frases_pos}
    lexico_frases.update({normalizar(f): -1.5 for f in frases_neg})

    lexico_palabras: Dict[str, float] = {normalizar(p): 1.0 for p in palabras_pos}
    lexico_palabras.update({normalizar(p): -1.0 for p in palabras_neg})

    return lexico_frases, lexico_palabras


def _contar_patron(texto: str, patron: str) -> int:
    """Cuenta ocurrencias no solapadas de un patrón literal como palabra/frase.

    Usa límites de palabra cuando es posible. Para frases con espacios, usa
    coincidencia literal con fronteras aproximadas.
    """
    patron_escapado = re.escape(patron)
    if " " in patron:
        regex = rf"(?<!\w){patron_escapado}(?!\w)"
    else:
        regex = rf"\b{patron_escapado}\b"
    return len(re.findall(regex, texto))


def score_sentimiento(texto: str, lexico_frases: Dict[str, float], lexico_palabras: Dict[str, float]) -> float:
    """Calcula un score de sentimiento simple combinando frases y palabras.

    Reglas:
      - Procesa primero frases (peso 1.5), luego palabras (peso 1.0).
      - Normaliza por longitud: score / sqrt(tokens + usados).

    Args:
        texto: Texto bruto.
        lexico_frases: Diccionario frase->peso (+/-1.5).
        lexico_palabras: Diccionario palabra->peso (+/-1.0).
    Returns:
        Score normalizado (float).
    """
    texto_norm = normalizar(texto)
    if not texto_norm:
        return 0.0

    score_bruto = 0.0
    usados = 0
    resto = texto_norm

    # Frases
    for frase, peso in lexico_frases.items():
        cnt = _contar_patron(resto, frase)
        if cnt > 0:
            score_bruto += cnt * peso
            usados += cnt
            # Evitar doble conteo eliminando las ocurrencias
            resto = re.sub(re.escape(frase), " ", resto)

    # Palabras
    tokens = [t for t in re.split(r"\W+", resto) if t]
    for token in tokens:
        if token in lexico_palabras:
            score_bruto += lexico_palabras[token]
            usados += 1

    num_tokens = len(tokens) if len(tokens) > 0 else len(re.split(r"\W+", texto_norm))
    denom = math.sqrt(max(1, num_tokens + usados))
    return float(score_bruto / denom) if denom > 0 else float(score_bruto)


def etiqueta(score: float, pos: float = 0.6, neg: float = -0.6) -> str:
    """Devuelve etiqueta textual de sentimiento a partir del score."""
    if score >= pos:
        return "positivo"
    if score <= neg:
        return "negativo"
    return "neutral"


def top_terminos(textos: Iterable[str], n: int = 15) -> pd.DataFrame:
    """Calcula términos más frecuentes excluyendo stopwords y tokens cortos (<=2)."""
    stopwords = {
        "a","acaso","ademas","al","algo","algun","alguna","algunas","alguno","algunos","alli","ambos","ante",
        "antes","aquel","aquella","aquellas","aquello","aquellos","aqui","asi","aun","aunque","cada","como",
        "con","contra","cual","cuales","cualquier","cuando","de","debe","debido","del","demas","desde","donde",
        "dos","el","ella","ellas","ello","ellos","en","entre","era","erais","eran","eras","eres","es","esa",
        "esas","ese","eso","esos","esta","estaba","estabais","estaban","estabas","estad","estada","estadas",
        "estado","estados","estais","estamos","estan","estar","estara","estas","este","estos","estoy","fin",
        "fue","fueron","fui","fuimos","ha","habeis","habia","habiais","habian","habias","han","has","hasta",
        "hay","la","las","le","les","lo","los","mas","me","mi","mis","mucho","muy","nada","ni","no","nos",
        "nosotros","nuestra","nuestras","nuestro","nuestros","nunca","o","os","otra","otros","para","pero","poco",
        "por","porque","que","quien","quienes","se","sea","segun","ser","seria","si","siempre","sin","sobre",
        "sois","solamente","solo","somos","soy","su","sus","tal","tambien","tampoco","tan","tanto","te","teneis",
        "tenemos","tener","tengo","ti","tiempo","tiene","tienen","toda","todas","todo","todos","tu","tus","un",
        "una","uno","unos","usa","usted","ustedes","va","vais","valor","vamos","van","varias","varios","vos",
        "vose","vuestra","vuestras","vuestro","vuestros","y","ya"
    }
    contador: Counter[str] = Counter()
    for texto in textos:
        t = normalizar(texto)
        # Remover URLs, menciones y hashtags básicos para mejor señal de términos
        t = re.sub(r"https?://\S+", " ", t)
        t = re.sub(r"[@#]\w+", " ", t)
        tokens = [tok for tok in re.split(r"\W+", t) if tok and len(tok) > 2 and tok not in stopwords]
        contador.update(tokens)
    items = contador.most_common(n)
    return pd.DataFrame(items, columns=["termino", "frecuencia"])


# ==========================
# Ingesta
# ==========================
def ingesta_sample() -> pd.DataFrame:
    """Crea un dataset de ejemplo (~30 filas) con polaridades variadas."""
    base_fecha = datetime.now()
    textos = [
        "El despliegue fue excelente, anda rapido en Kubernetes",
        "La atencion pesima del soporte, no lo recomiendo",
        "Pipeline de CI/CD muy bueno, super recomendado",
        "Autoscaling anda lento hoy, mal servicio",
        "SRE crack resolviendo un bug critico, fix perfecto",
        "No me gusta el cambio, horrible performance",
        "Migracion vale la pena, costo barato y anda rapido",
        "No vale la pena mover este monolito, demasiado caro",
        "Monitorizacion genial, me encanta la visibilidad",
        "Alertas terribles, ruido y falso positivo",
        "Infra as Code top, excelente soporte del equipo",
        "La latencia es mala, anda lento el balanceador",
        "Backups listo, me encanta la automatizacion",
        "Oncall triste hoy, estafa en el proveedor",
        "El cluster anda rapido, escalado muy bueno",
        "No funciona el despliegue azul/verde",
        "El costo es caro, rendimiento lento",
        "Documentacion excelente, super recomendado",
        "No lo recomiendo, muy malo el servicio",
        "Feature toggle genial, deploy sin downtime",
        "Bug terrible en la version nueva",
        "Fix rapido del incidente, feliz con el equipo",
        "Observabilidad top, amor por los dashboards",
        "La base anda lento, query horrible",
        "Me encanta el nuevo pipeline, vale la pena",
        "Atencion pesima del proveedor cloud",
        "Horrible experiencia con el rollback, odio esto",
        "Excelente soporte post mortem, muy bueno",
        "Monitoreo barato pero malo",
        "Buena practica, crack el equipo de DevOps",
    ]
    registros: List[Dict[str, str]] = []
    for i, txt in enumerate(textos):
        fecha = (base_fecha - timedelta(days=(len(textos) - i))).strftime("%Y-%m-%d")
        registros.append(
            {
                "id": f"sample_{i+1}",
                "fecha": fecha,
                "autor": f"user{i%7}",
                "texto": txt,
                "tema": "DevOps",
            }
        )
    return pd.DataFrame(registros)


def ingesta_snscrape(query: str, desde: str, limite: int) -> pd.DataFrame:
    """Ingesta con snscrape vía CLI, parseando JSONL.

    Requiere snscrape instalado en el PATH. Si no está, se informa el comando
    de instalación y termina con código 1.
    """
    comando = [
        "snscrape",
        "--jsonl",
        "--max-results",
        str(limite),
        "twitter-search",
        f"{query} lang:es since:{desde}",
    ]
    try:
        res = subprocess.run(
            comando,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        print("[ERROR] snscrape no está instalado. Instalá con `pip install snscrape`.")
        sys.exit(1)

    if res.returncode != 0:
        print("[ERROR] snscrape devolvió un error. Ajustá query/fecha/limite y reintentá.")
        stderr = (res.stderr or "").strip()
        if stderr:
            print(stderr)
        sys.exit(1)

    lineas = [ln for ln in (res.stdout or "").splitlines() if ln.strip()]
    if not lineas:
        print("[INFO] snscrape no devolvió resultados. Probá ajustar query/fecha/limite.")
        return pd.DataFrame(columns=["id", "fecha", "autor", "texto", "tema"])

    filas: List[Dict[str, str]] = []
    for ln in lineas:
        try:
            obj = json.loads(ln)
            filas.append(
                {
                    "id": str(obj.get("id", "")),
                    "fecha": str(obj.get("date", ""))[:10],
                    "autor": (obj.get("user", {}) or {}).get("username", ""),
                    "texto": obj.get("content", ""),
                    "tema": "",
                }
            )
        except json.JSONDecodeError:
            continue

    df = pd.DataFrame(filas)
    # Asegurar columnas mínimas
    return df[["id", "fecha", "autor", "texto", "tema"]]


def ingesta_csv(csv_in: str) -> pd.DataFrame:
    """Lee CSV con columnas mínimas (fecha, autor, texto). 'tema' opcional. Crea 'id' si falta.
    Intenta normalizar 'fecha' a YYYY-MM-DD cuando sea factible.
    """
    if not os.path.exists(csv_in):
        print(f"[ERROR] No existe el archivo: {csv_in}")
        sys.exit(1)
    df = pd.read_csv(csv_in)
    columnas = set(df.columns.str.lower())
    requeridas = {"fecha", "autor", "texto"}
    if not requeridas.issubset(columnas):
        print("[ERROR] CSV debe incluir columnas: fecha, autor, texto. 'tema' es opcional.")
        sys.exit(1)

    # Normalizar nombres de columnas a minúsculas uniformes
    df = df.rename(columns={c: c.lower() for c in df.columns})
    if "id" not in df.columns:
        df.insert(0, "id", [f"csv_{i+1}" for i in range(len(df))])
    if "tema" not in df.columns:
        df["tema"] = ""

    # Normalizar fecha si es posible
    try:
        fechas = pd.to_datetime(df["fecha"], errors="coerce")
        df["fecha"] = fechas.dt.strftime("%Y-%m-%d").fillna(df["fecha"].astype(str))
    except Exception:
        pass

    return df[["id", "fecha", "autor", "texto", "tema"]]


# ==========================
# Pipeline
# ==========================
def procesar(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega score y sentimiento al DataFrame dado."""
    df_proc = df.copy()
    lexico_frases, lexico_palabras = construir_lexico()
    df_proc["score"] = df_proc["texto"].astype(str).apply(
        lambda t: score_sentimiento(t, lexico_frases, lexico_palabras)
    )
    df_proc["sentimiento"] = df_proc["score"].apply(etiqueta)
    return df_proc


def kpis(df: pd.DataFrame) -> Dict[str, float | int]:
    """Calcula KPIs básicos sobre el DataFrame procesado."""
    total = int(len(df))
    positivos = int((df["sentimiento"] == "positivo").sum())
    negativos = int((df["sentimiento"] == "negativo").sum())
    neutrales = int((df["sentimiento"] == "neutral").sum())
    promedio = float(df["score"].mean()) if total else 0.0
    mediana = float(df["score"].median()) if total else 0.0
    return {
        "total": total,
        "positivos": positivos,
        "negativos": negativos,
        "neutrales": neutrales,
        "promedio": promedio,
        "mediana": mediana,
    }


def graficos(df: pd.DataFrame, outdir: str) -> Tuple[str, str]:
    """Genera dos gráficos (barras por sentimiento y histograma de score)."""
    os.makedirs(outdir, exist_ok=True)
    barras_path = os.path.join(outdir, "sentimientos_barras.png")
    hist_path = os.path.join(outdir, "scores_hist.png")

    # Barras: distribución de sentimientos, orden: negativo, neutral, positivo
    orden = ["negativo", "neutral", "positivo"]
    conteos = [int((df["sentimiento"] == s).sum()) for s in orden]
    plt.figure()
    plt.bar(orden, conteos)
    plt.title("Distribución de sentimientos")
    plt.xlabel("Sentimiento")
    plt.ylabel("Cantidad")
    plt.tight_layout()
    plt.savefig(barras_path)
    plt.close()

    # Histograma de scores
    plt.figure()
    plt.hist(df["score"].values, bins=20)
    plt.title("Histograma de scores")
    plt.xlabel("Score")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(hist_path)
    plt.close()

    return barras_path, hist_path


def guardar(df: pd.DataFrame, outdir: str, top_n: int = 15) -> Tuple[str, str]:
    """Guarda CSV de posts con sentimiento y top términos."""
    os.makedirs(outdir, exist_ok=True)
    posts_path = os.path.join(outdir, "posts_con_sentimiento.csv")
    top_path = os.path.join(outdir, "top_terminos.csv")

    columnas = ["id", "fecha", "autor", "texto", "tema", "score", "sentimiento"]
    faltantes = [c for c in columnas if c not in df.columns]
    if faltantes:
        for c in faltantes:
            df[c] = "" if c != "score" and c != "sentimiento" else 0
    df[columnas].to_csv(posts_path, index=False, encoding="utf-8")

    top_df = top_terminos(df["texto"].astype(str).tolist(), n=top_n)
    top_df.to_csv(top_path, index=False, encoding="utf-8")

    return posts_path, top_path


# ==========================
# CLI
# ==========================
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Análisis de sentimiento ES (X/Twitter) sin API oficial."
    )
    parser.add_argument(
        "--modo",
        choices=["sample", "sn", "csv"],
        default="sample",
        help="Fuente de datos: sample | sn | csv",
    )
    parser.add_argument(
        "--query",
        default="DevOps Argentina",
        help="Consulta para snscrape (solo --modo sn)",
    )
    parser.add_argument(
        "--desde",
        default="2025-08-01",
        help="Fecha desde (YYYY-MM-DD) para snscrape (solo --modo sn)",
    )
    parser.add_argument(
        "--limite",
        type=int,
        default=200,
        help="Máximo de resultados para snscrape (solo --modo sn)",
    )
    parser.add_argument(
        "--csv_in",
        default="",
        help="Ruta al CSV de entrada (solo --modo csv)",
    )
    parser.add_argument(
        "--outdir",
        default="salida",
        help="Directorio de salida para CSV/PNG",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Ingesta según modo
    if args.modo == "sample":
        df = ingesta_sample()
    elif args.modo == "sn":
        df = ingesta_snscrape(args.query, args.desde, args.limite)
        if df.empty:
            # Ya se informó que no hubo resultados
            sys.exit(0)
    else:  # csv
        if not args.csv_in:
            print("[ERROR] Debe indicar --csv_in para --modo csv")
            print("Uso: python app.py --modo csv --csv_in ruta/al/archivo.csv")
            sys.exit(2)
        df = ingesta_csv(args.csv_in)

    # Procesamiento
    df_proc = procesar(df)

    # KPIs
    indicadores = kpis(df_proc)

    # Salidas
    posts_csv, top_csv = guardar(df_proc, args.outdir)
    barras_png, hist_png = graficos(df_proc, args.outdir)

    # Consola
    print("== KPIs ==")
    print(
        f"Total: {indicadores['total']} | +: {indicadores['positivos']} | -: {indicadores['negativos']} | 0: {indicadores['neutrales']}"
    )
    print(
        f"Score promedio: {indicadores['promedio']:.4f} | mediana: {indicadores['mediana']:.4f}"
    )
    print("== Artefactos ==")
    print(posts_csv)
    print(top_csv)
    print(barras_png)
    print(hist_png)


if __name__ == "__main__":
    main()


