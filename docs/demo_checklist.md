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
  `uv run python -m src.models.numpy_nn.gradcheck` prints the relative error of every analytic
  gradient against a numerical one (worst ~1e-9; anything below 1e-6 passes).
- **NumPy vs PyTorch on real data:** `uv run python -m src.models.torch_nn` trains both networks
  from the same weights (~20 s) and prints the differences (predictions ~1e-14 µg/m³). The saved
  result and loss-curve figure are in `results/numpy_vs_torch/` if there is no time to rerun.
- **Expert-system trace:**
  `uv run python -m src.expert --pm25 85 --today 60 --sensitive --outdoor` prints the advice
  for a sample forecast plus which rules fired and why (vary `--pm25`, `--today`,
  `--sensitive`, `--outdoor` to show different bands/rules).
