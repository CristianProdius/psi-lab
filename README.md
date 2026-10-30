# PASI — Lucrarea de laborator N1 (Python)

Modelarea semnalelor elementare şi teorema eşantionării, portate din MATLAB în Python (NumPy / SciPy / Matplotlib).

| Fişier | Rol |
|---|---|
| `lab1.ipynb` | Notebook-ul executat (cod + grafice + concluzii) |
| `lab1.html` | Acelaşi raport, exportat pentru predare / vizualizare în browser |
| `build_notebook.py` | Sursa notebook-ului (regenerează `lab1.ipynb`) |

## Rulare

```bash
uv sync                           # instalează dependenţele din uv.lock
uv run jupyter lab lab1.ipynb     # deschide interactiv
```

Regenerare + execuţie + export:

```bash
uv run python build_notebook.py
uv run jupyter nbconvert --execute --inplace --to notebook lab1.ipynb
uv run jupyter nbconvert --to html lab1.ipynb
```
