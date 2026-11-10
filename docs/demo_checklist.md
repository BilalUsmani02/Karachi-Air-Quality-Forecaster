# Pre-demo checklist

Short run-through for the morning of a demo or viva.

## The morning of the demo

- [ ] `uv run python -m src.live --refresh` — fetches today's readings and weather and builds a
  fresh snapshot. Needs internet and `OPENAQ_API_KEY` in `.env`; takes about 2.5 minutes
  (OpenAQ's request rate limit).
- [ ] Pre-open every app page once: `uv run streamlit run app/main.py`, then click through
  Forecast, Model comparison and About & data. First load is slow (~5 s, scikit-learn
  import) and the SHAP reasons section is slow the first time it runs (~15 s) — both are fast
  on every visit after that, so do this before anyone is watching.
- [ ] Build (or copy onto the laptop) a fresh demo kit: `uv run python scripts/make_demo_kit.py`
  writes `dist/demo_kit_<date>.zip`. Keep that zip on a **USB drive** as a backup — see
  "If the internet is down" below for how to use it.

## Do NOT do on demo day

- [ ] **Never run `uv sync` or upgrade any package.** `models/final_ensemble.pkl` is a Python
  pickle saved with the exact NumPy / scikit-learn / PyTorch versions in `uv.lock`; a newer
  library version can fail to load it, or load it and silently behave differently. If you need
  a clean environment on another machine, use `uv sync --frozen` (installs exactly what
  `uv.lock` pins, changes nothing) — never a bare `uv sync`.

## If the internet is down

- Use the **Demo mode toggle in the sidebar** (app) to show a stored historical day instead of
  a live forecast — the app says so with a banner, so it's clear this isn't a live reading.
- If the laptop itself needs rebuilding: copy `dist/demo_kit_<date>.zip` from the USB drive,
  unzip it into the project root (it restores `models/`, `data/processed/`, `data/cache/` and
  `uv.lock` to their normal places), run `uv sync --frozen`, then
  `uv run streamlit run app/main.py`. No internet or API key needed — the app falls back to
  the cached/demo snapshot automatically.

## If someone asks to see the internals

- **From-scratch network + gradient check:**
  `uv run pytest tests/test_numpy_nn_layers.py -v` runs the gradient-check tests (the NumPy
  network's analytic gradients checked against numerical ones) and prints each one as it
  passes. `uv run pytest tests/test_torch_nn.py -v` shows the NumPy network training
  identically to its PyTorch twin from the same initial weights (same loss, matching
  predictions).
- **Expert-system trace:**
  `uv run python -m src.expert --pm25 85 --today 60 --sensitive --outdoor` prints the advice
  for a sample forecast plus which rules fired and why (vary `--pm25`, `--today`,
  `--sensitive`, `--outdoor` to show different bands/rules).
