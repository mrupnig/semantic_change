# Grammatical Profiling for Semantic Change Detection

Dieses Projekt repliziert das Verfahren des **Grammatical Profiling** nach Giulianelli, Kutuzov & Pivovarova (2021) auf einem eigenen historischen Textkorpus. Untersucht wird semantischer Wandel englischsprachiger Wörter im Zeitraum 1800–1920, indem grammatische Profile (POS-Tags, morphologische Kategorien, syntaktische Abhängigkeitsrelationen) zweier Zeitscheiben miteinander verglichen werden. Grundlage ist die Idee, dass semantischer Wandel Spuren im morphosyntaktischen Verhalten von Wörtern hinterlässt – messbar als Cosinus-Distanz zwischen Frequenzvektoren.

**Referenzpaper:** Giulianelli, M., Kutuzov, A., & Pivovarova, L. (2021). *Grammatical Profiling for Semantic Change Detection*. Proceedings of CoNLL 2021, S. 423–434. https://aclanthology.org/2021.conll-1.33.pdf

---

## Projektstruktur

```
.
├── data/                   # Textkorpus (.txt-Dateien) und metadata.csv
├── notebooks/
│   └── analysis.ipynb      # Interaktive Auswertung (Haupteinstiegspunkt)
├── src/semantic_change/
│   ├── corpus.py           # Laden und Filtern der Texte
│   ├── profiling.py        # Grammatical Profiling (spaCy)
│   ├── distance.py         # Cosinus-Distanz und Ranking
│   └── visualization.py    # Plots und Heatmaps
├── output/                 # Generierte Plots und Tabellen (nach Ausführung)
├── main.py                 # CLI-Einstiegspunkt
└── pyproject.toml
```

---

## Voraussetzungen

- Python 3.12 (spaCy ist mit Python ≥ 3.14 nicht kompatibel)
- [uv](https://docs.astral.sh/uv/) als Paketmanager

---

## Installation

```bash
# Repository klonen
git clone <repo-url>
cd semantic_change

# Virtuelle Umgebung erstellen und Abhängigkeiten installieren
uv sync
```

Das spaCy-Sprachmodell `en_core_web_sm` wird automatisch mit installiert (via `uv.sources` in `pyproject.toml`).

---

## Anwendung

### Jupyter Notebook (empfohlen)

```bash
uv run jupyter notebook notebooks/analysis.ipynb
```

Am Anfang des Notebooks können Parameter angepasst werden:

| Parameter | Standard | Bedeutung |
|---|---|---|
| `MAX_OCCURRENCES` | `50` | Maximale Belegstellen pro Wort und Zeitscheibe |
| `SEED` | `42` | Zufalls-Seed für reproduzierbare Stichprobenziehung |
| `SAVE_PLOTS` | `True` | Plots als PNG in `output/` speichern |
| `SAVE_TABLES` | `True` | Ergebnistabellen als CSV in `output/` speichern |
| `OUTPUT_DIR` | `"../output"` | Ausgabeverzeichnis |

Das Notebook führt die gesamte Analyse durch: Korpusstatistiken, Profilberechnung, Ranking, Heatmap, Einzelwortanalysen und Ergebnistabellen für das Paper.

### Kommandozeile

```bash
uv run python main.py
uv run python main.py --max-occurrences 50
```

Ergebnisse werden in `output/` gespeichert.

---

## Zielwörter

Die Analyse umfasst 14 Zielwörter, die thematisch zum Untersuchungszeitraum (Industrialisierung, Wissenschaft, Kolonialismus) passen:

```
steam, electric, race, progress, science, machine,
labour, civilise, industry, telegraph, evolution,
empire, reform, capital, engine
```

---

## Zeitscheiben

| Zeitscheibe | Zeitraum |
|---|---|
| T1 | 1800–1860 |
| T2 | 1861–1920 |
