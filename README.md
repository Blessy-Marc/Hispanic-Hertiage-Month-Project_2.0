# Hispanic Heritage Historical Data Explorer

A self-contained Streamlit classroom project using historical indicators from the World Bank, polynomial regression, custom functions, and introductory calculus analysis.

## Files
- `app.py` — complete Streamlit application
- `requirements.txt` — minimal, unpinned dependencies
- `README.md` — setup, deployment, and verification
- `sources.md` — indicator codes, definitions, and units

## Run locally
Python 3.10 or newer is recommended. From the project folder:

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

Internet access is needed for World Bank API retrieval.

## Streamlit Community Cloud
1. Create a GitHub repository.
2. Upload all four files to the repository root.
3. Visit https://share.streamlit.io/ and connect GitHub.
4. Select the repository and branch; set the main file to `app.py`.
5. Deploy. Review app logs for dependency, startup, or API errors.

## Render
Create a Python Web Service linked to the repository.

Build command:
```bash
pip install -r requirements.txt
```

Start command (single line):
```bash
python -m streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT
```

## Verification checklist
- Check country, indicator code, definition, units, API URL, retrieval date, observation count, and actual coverage.
- Test source-data reset, edited values, non-integer years, duplicate years, null values, and indicator-specific constraints.
- Test degrees 3–8 and insufficient distinct-year counts.
- Inspect centered-time equation, base year, in-sample R-squared, in-sample MAE, residuals, and chronological holdout score when available.
- Check single-year prediction against the same year in batch output.
- Test batch ranges, step sizes 1–10, comma-separated years, malformed input, and years outside 1950–2100.
- Change comparison countries, graph sample increment, forward horizon, and printer-friendly setting.
- Try each context and function family; test log/rational domain restrictions, derivative toggles, tangent and secant lines, and interval controls.
- Open CSV, XLSX, and PDF exports and check their contents.
- Simulate API failure by disconnecting network access; ensure the app reports the error and does not invent historical observations.
- Deploy and repeat key checks in the hosted environment.

## Troubleshooting
- `ModuleNotFoundError`: install dependencies again and inspect deployment logs.
- API timeout/HTTP error: retry later, verify internet access and the API URL.
- Not enough observations: lower polynomial degree or choose a country/indicator with sufficient coverage.
- Function parse error: use `x`, numeric constants, parameters `a`–`d`, arithmetic, and supported functions (`sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `Abs`).
- Undefined function values: check logarithm and denominator restrictions; gaps in the graph can be genuine domain gaps.
- Render startup error: confirm Python Web Service, repository root, `app.py`, and the `$PORT` start command.

## Mathematical notes and limitations
Regression uses centered time `t = year - base year`; predictions use the same centering. Degree `d` requires at least `d+1` distinct valid years. R² and MAE are in-sample measures; a chronological holdout score is separately labeled when there are enough observations. A polynomial fit is descriptive, not causal. Extrapolation can become implausible. Residuals and MAE communicate model error but are not a validated 95% prediction interval.

The function lab uses SymPy for symbolic differentiation and numerical sampling for graphs and root candidates. A numerical scan may miss repeated or closely spaced roots and may not fully characterize all discontinuities. A zero of the first derivative is not automatically an extremum; a zero of the second derivative is not automatically an inflection point. Contextual functions are educational models, not validated medical, engineering, weather, or economic forecasts. The simulation display is normalized for classroom visualization and is not a calibrated physical/medical simulation.

## Runtime verification statement
The source is syntax-checked when packaged. This does not establish a successful Streamlit browser session, live World Bank response, or end-to-end export verification. Complete the checklist locally and in deployment before relying on the app for a final presentation.
