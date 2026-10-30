"""
Hispanic Heritage Historical Data Explorer
Self-contained Streamlit classroom application.
"""
from __future__ import annotations

import io
import math
import re
from datetime import date, datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import sympy as sp
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Hispanic Heritage Historical Data Explorer", page_icon="🌎", layout="wide")

COUNTRIES = {
    "Mexico": "MEX", "Brazil": "BRA", "Argentina": "ARG", "Colombia": "COL",
    "Chile": "CHL", "Peru": "PER", "Venezuela": "VEN", "Ecuador": "ECU", "Uruguay": "URY"
}
INDICATORS = {
    "Life expectancy at birth": ("SP.DYN.LE00.IN", "years", "Average number of years a newborn would live if current mortality patterns persisted."),
    "Unemployment, national estimate": ("SL.UEM.TOTL.NE.ZS", "% of labor force", "Share of the labor force without work but available for and seeking employment, using national estimates."),
    "GDP per capita": ("NY.GDP.PCAP.CD", "current US dollars per person", "Gross domestic product divided by midyear population. It is not average wealth or household income."),
    "Crude birth rate": ("SP.DYN.CBRT.IN", "births per 1,000 people per year", "Annual number of live births per 1,000 people. This is a rate, not a birth count."),
    "Net migration": ("SM.POP.NETM", "people", "Net number of migrants over the period; it is not an emigration count."),
    "Intentional homicide rate": ("VC.IHR.PSRC.P5", "per 100,000 people", "Intentional homicides per 100,000 population."),
    "Consumer price inflation": ("FP.CPI.TOTL.ZG", "% annual", "Annual percentage change in the consumer price index."),
    "Central government debt": ("GC.DOD.TOTL.GD.ZS", "% of GDP", "Central government debt as a percentage of GDP."),
    "Total reserves including gold": ("FI.RES.TOTL.CD", "current US dollars", "Total reserves including gold, expressed in current US dollars."),
    "External balance on goods and services": ("NE.RSB.GNFS.ZS", "% of GDP", "External balance on goods and services as a percentage of GDP.")
}
CONTEXTS = {
    "Business": ("Revenue", "US dollars", "time", "years", "Revenue or cost trend; illustrative unless fitted to data."),
    "Engineering": ("Displacement", "meters", "time", "seconds", "Position/displacement; first derivative is velocity and second is acceleration."),
    "Medical": ("Concentration", "mg/L", "time", "hours", "Illustrative concentration curve, not medical advice or a dosing model."),
    "Sports science": ("Performance", "score", "time", "weeks", "Illustrative performance measure over training time."),
    "Biology": ("Population", "organisms", "time", "days", "Illustrative population or cell-colony growth."),
    "Bioengineering": ("Signal", "arbitrary units", "time", "seconds", "Illustrative biological signal."),
    "Economic forecasting": ("Indicator", "indicator units", "time", "years", "Illustrative economic trend; not a validated forecast."),
    "Weather": ("Temperature", "°C", "time", "hours", "Illustrative temperature cycle, not a weather forecast."),
    "Custom": ("Output", "units", "input", "units", "Student-defined relationship.")
}
FUNCTION_PRESETS = {
    "Quadratic": "a*x**2 + b*x + c",
    "Cubic": "a*x**3 + b*x**2 + c*x + d",
    "Rational": "a/(x-h) + k",
    "Exponential": "a*exp(b*x) + c",
    "Logarithmic": "a*log(x-h) + c",
    "Trigonometric": "a*sin(b*x) + c"
}
ALLOWED_FUNCS = {"sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt, "Abs": sp.Abs}
x = sp.Symbol("x", real=True)

def fetch_world_bank(country: str, indicator: str, start_year: int, end_year: int) -> tuple[pd.DataFrame, str]:
    url = f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"
    params = {"date": f"{start_year}:{end_year}", "format": "json", "per_page": 20000}
    response = requests.get(url, params=params, timeout=25)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
        return pd.DataFrame(columns=["year", "value"]), response.url
    rows = [{"year": int(r["date"]), "value": r["value"]} for r in payload[1] if r.get("date")]
    frame = pd.DataFrame(rows, columns=["year", "value"])
    if not frame.empty:
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        frame = frame.dropna(subset=["value"]).sort_values("year").reset_index(drop=True)
    return frame, response.url

@st.cache_data(ttl=3600, show_spinner=False)
def cached_fetch(country: str, indicator: str, start_year: int, end_year: int):
    return fetch_world_bank(country, indicator, start_year, end_year)

def validate_data(frame: pd.DataFrame, indicator_name: str) -> tuple[pd.DataFrame, list[str]]:
    errors = []
    if not {"year", "value"}.issubset(frame.columns):
        return pd.DataFrame(columns=["year", "value"]), ["Required columns are year and value."]
    out = frame[["year", "value"]].copy()
    numeric_year = pd.to_numeric(out["year"], errors="coerce")
    numeric_value = pd.to_numeric(out["value"], errors="coerce")
    if numeric_year.isna().any() or ((numeric_year % 1) != 0).any():
        errors.append("Every year must be an integer.")
    if numeric_value.isna().any():
        errors.append("Every value must be numeric and non-null.")
    out["year"], out["value"] = numeric_year, numeric_value
    out = out.dropna()
    if out["year"].duplicated().any():
        errors.append("Duplicate years are not allowed.")
    if indicator_name == "Life expectancy at birth" and ((out.value < 0) | (out.value > 120)).any():
        errors.append("Life expectancy must be between 0 and 120 years.")
    if indicator_name in ("Unemployment, national estimate", "Central government debt") and (out.value < 0).any():
        errors.append("This indicator cannot be negative.")
    if indicator_name == "Intentional homicide rate" and (out.value < 0).any():
        errors.append("A homicide rate cannot be negative.")
    if indicator_name == "Crude birth rate" and (out.value < 0).any():
        errors.append("A crude birth rate cannot be negative.")
    return out.sort_values("year").reset_index(drop=True), errors

def poly_fit(frame: pd.DataFrame, degree: int):
    clean, errors = validate_data(frame, st.session_state.indicator_name)
    if errors:
        raise ValueError(" ".join(errors))
    if clean.year.nunique() < degree + 1:
        raise ValueError(f"Degree {degree} requires at least {degree+1} distinct valid years; found {clean.year.nunique()}.")
    base = int(clean.year.min())
    t = clean.year.to_numpy(dtype=float) - base
    coeff = np.polyfit(t, clean.value.to_numpy(dtype=float), degree)
    model = np.poly1d(coeff)
    fitted = model(t)
    return base, coeff, model, fitted, clean

def safe_expression(source: str, parameter_values: dict[str, float]):
    if len(source) > 250:
        raise ValueError("Expression is too long.")
    if not re.fullmatch(r"[0-9A-Za-z_+\-*/()., \t**]+", source):
        raise ValueError("Use numbers, x, parameters a–h, arithmetic operators, and supported functions only.")
    local = {"x": x, **ALLOWED_FUNCS, **{k: sp.Float(v) for k, v in parameter_values.items()}}
    try:
        expr = sp.sympify(source, locals=local, evaluate=True)
    except Exception as exc:
        raise ValueError(f"Could not parse expression: {exc}") from exc
    if expr.free_symbols - {x}:
        raise ValueError("Expression contains an unknown variable or parameter.")
    if expr.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        raise ValueError("Expression contains an undefined or infinite constant.")
    return sp.simplify(expr)

def numerical(expr, values):
    fn = sp.lambdify(x, expr, modules=["numpy"])
    with np.errstate(all="ignore"):
        result = np.asarray(fn(values), dtype=float)
    return result

def derivative_report(expr, left: float, right: float):
    d1, d2 = sp.diff(expr, x), sp.diff(expr, x, 2)
    points = np.linspace(left, right, 1601)
    y = numerical(expr, points)
    dy = numerical(d1, points)
    ddy = numerical(d2, points)
    finite = np.isfinite(y)
    # Numeric sign-change root candidates; exact symbolic forms are displayed separately.
    def roots(vals):
        found = []
        for i in range(len(points)-1):
            if np.isfinite(vals[i:i+2]).all() and vals[i] == 0:
                found.append(points[i])
            elif np.isfinite(vals[i:i+2]).all() and vals[i] * vals[i+1] < 0:
                found.append((points[i] + points[i+1]) / 2)
        return [round(v, 5) for j, v in enumerate(found) if j == 0 or abs(v-found[j-1]) > (right-left)/500]
    critical = roots(dy)
    inflect_candidates = roots(ddy)
    inflections = []
    for p in inflect_candidates:
        eps = max((right-left)/10000, 1e-5)
        before, after = numerical(d2, np.array([p-eps, p+eps]))
        if np.isfinite([before, after]).all() and before*after < 0:
            inflections.append(p)
    return d1, d2, points, y, dy, ddy, critical, inflections

def make_pdf(title, details, chart_bytes, preview, calc):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=32, leftMargin=32, topMargin=32, bottomMargin=32)
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Spacer(1, 8)]
    for line in details:
        story.append(Paragraph(str(line).replace("&", "&amp;"), styles["BodyText"]))
        story.append(Spacer(1, 4))
    if chart_bytes:
        story += [Spacer(1, 8), RLImage(io.BytesIO(chart_bytes), width=500, height=280)]
    story += [Spacer(1, 8), Paragraph("Historical data preview", styles["Heading2"])]
    if not preview.empty:
        data = [list(preview.columns)] + preview.head(12).astype(str).values.tolist()
        table = Table(data, repeatRows=1)
        table.setStyle(TableStyle([("BACKGROUND", (0,0),(-1,0),colors.HexColor("#e8edf3")),("GRID",(0,0),(-1,-1),0.4,colors.grey),("FONTSIZE",(0,0),(-1,-1),7)]))
        story.append(table)
    if not calc.empty:
        story += [Spacer(1, 8), Paragraph("Calculated years", styles["Heading2"])]
        data = [list(calc.columns)] + calc.head(20).astype(str).values.tolist()
        table = Table(data, repeatRows=1)
        table.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.4,colors.grey),("FONTSIZE",(0,0),(-1,-1),7)]))
        story.append(table)
    doc.build(story)
    return buffer.getvalue()

st.title("🌎 Hispanic Heritage Historical Data Explorer")
st.caption("HISPANIC HERITAGE DATA LAB • Historical evidence, mathematical modeling, and interactive function analysis")
st.info("Research question: customize this in your project notes. Brazil is included as a Latin American comparison; it is not a Spanish-speaking country.")

with st.sidebar:
    st.header("Historical data")
    country_name = st.selectbox("Country", list(COUNTRIES), index=0)
    indicator_name = st.selectbox("Indicator", list(INDICATORS), index=0)
    degree = st.slider("Polynomial degree", 3, 8, 3)
    sample_step = st.slider("Graph sample increment (years)", 1, 10, 2)
    horizon = st.slider("Forward display horizon (years)", 0, 50, 10)
    compare_names = st.multiselect("Comparison countries", [c for c in COUNTRIES if c != country_name], default=[])
    printer_mode = st.checkbox("Printer-friendly view", False)

current_year = date.today().year
start_year, end_year = current_year - 70, current_year - 1
indicator_code, units, definition = INDICATORS[indicator_name]
st.session_state.setdefault("indicator_name", indicator_name)
st.session_state.indicator_name = indicator_name

try:
    with st.spinner("Retrieving World Bank observations…"):
        source_data, api_url = cached_fetch(COUNTRIES[country_name], indicator_code, start_year, end_year)
    if source_data.empty:
        st.warning("The API returned no non-null observations for this country and indicator in the requested period.")
    else:
        st.success(f"Retrieved {len(source_data)} observations. Actual coverage: {int(source_data.year.min())}–{int(source_data.year.max())}.")
except requests.Timeout:
    source_data, api_url = pd.DataFrame(columns=["year","value"]), ""
    st.error("World Bank request timed out. Try again later.")
except requests.RequestException as exc:
    source_data, api_url = pd.DataFrame(columns=["year","value"]), ""
    st.error(f"World Bank request failed: {exc}")
except Exception as exc:
    source_data, api_url = pd.DataFrame(columns=["year","value"]), ""
    st.error(f"Could not process the World Bank response: {exc}")

meta1, meta2, meta3 = st.columns(3)
meta1.metric("Requested years", f"{start_year}–{end_year}")
meta2.metric("Observations", len(source_data))
meta3.metric("Retrieved", datetime.now().strftime("%Y-%m-%d %H:%M"))
st.write(f"**Definition:** {definition}  \n**Units:** {units}  \n**Source:** [World Bank indicator API]({api_url or 'https://data.worldbank.org/'})  \n**Indicator code:** `{indicator_code}`")

if "source_snapshot" not in st.session_state or st.session_state.get("source_key") != (country_name, indicator_name):
    st.session_state.source_snapshot = source_data.copy()
    st.session_state.working_data = source_data.copy()
    st.session_state.source_key = (country_name, indicator_name)
    st.session_state.modified = False
if st.button("Reset to source data"):
    st.session_state.working_data = st.session_state.source_snapshot.copy()
    st.session_state.modified = False
    st.rerun()

st.subheader("Validated editable data")
st.caption("Edit the year/value table. The original API response remains separate. Changes are treated as a user-modified scenario.")
edited = st.data_editor(st.session_state.working_data, num_rows="dynamic", use_container_width=True, key=f"editor_{country_name}_{indicator_code}")
validated, validation_errors = validate_data(edited, indicator_name)
if not edited.equals(st.session_state.working_data):
    st.session_state.working_data = edited.copy()
    st.session_state.modified = True
if validation_errors:
    for err in validation_errors:
        st.error(err)
else:
    st.success("Working table passes the current validation rules." + (" User-modified scenario." if st.session_state.modified else " Source data unchanged."))

fit_frame = validated
fit_result = None
if len(fit_frame) >= degree + 1 and not validation_errors:
    try:
        fit_result = poly_fit(fit_frame, degree)
    except Exception as exc:
        st.error(f"Regression unavailable: {exc}")
else:
    st.warning(f"Regression requires at least {degree+1} distinct valid years; currently {fit_frame.year.nunique() if not fit_frame.empty else 0}.")

calc_df = pd.DataFrame(columns=["year","estimate","units","classification"])
equation = "Not fitted"
r2 = mae = np.nan
chart_bytes = None
fig = None
if fit_result:
    base, coeff, model, fitted, clean = fit_result
    residuals = clean.value.to_numpy() - fitted
    r2 = r2_score(clean.value, fitted) if len(clean) > 1 else np.nan
    mae = mean_absolute_error(clean.value, fitted)
    poly_terms = []
    deg = len(coeff)-1
    for i, c in enumerate(coeff):
        power = deg-i
        poly_terms.append(f"{c:.8g}·t^{power}" if power > 1 else (f"{c:.8g}·t" if power == 1 else f"{c:.8g}"))
    equation = "y = " + " + ".join(poly_terms) + f", where t = year − {base}"
    st.subheader("Polynomial model")
    st.code(equation)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Base year", base)
    c2.metric("Degree", degree)
    c3.metric("In-sample R²", f"{r2:.5f}")
    c4.metric("In-sample MAE", f"{mae:.5g} {units}")
    st.write(f"**Observation count:** {len(clean)} · **Historical range:** {clean.year.min()}–{clean.year.max()} · **Historical mean:** {clean.value.mean():.6g} {units}")
    with st.expander("Residuals and chronological holdout"):
        residual_table = clean.assign(fitted=fitted, residual=residuals)
        st.dataframe(residual_table, use_container_width=True)
        hold_n = max(1, int(round(len(clean)*0.2)))
        if len(clean) >= degree + 1 + hold_n:
            train = clean.iloc[:-hold_n]
            test = clean.iloc[-hold_n:]
            if train.year.nunique() >= degree+1:
                hbase = int(train.year.min())
                hmodel = np.poly1d(np.polyfit(train.year.to_numpy()-hbase, train.value.to_numpy(), degree))
                pred = hmodel(test.year.to_numpy()-hbase)
                st.write(f"Chronological holdout MAE (separate from in-sample): {mean_absolute_error(test.value, pred):.6g} {units}; test observations: {len(test)}.")
        else:
            st.caption("Not enough observations for a separate chronological holdout at this degree.")
    # Single-year prediction state
    st.subheader("Calculate a single year")
    single_year = st.number_input("Prediction year", min_value=1950, max_value=2100, value=min(max(current_year,1950),2100), step=1)
    if st.button("Calculate Value", key="single_calc"):
        st.session_state.single_result = (country_name, indicator_name, degree, tuple(clean.year), int(single_year), float(model(int(single_year)-base)))
    key_now = (country_name, indicator_name, degree, tuple(clean.year))
    if st.session_state.get("single_result") and st.session_state.single_result[:4] == key_now:
        saved = st.session_state.single_result
        cls = "within fitted historical range" if clean.year.min() <= saved[4] <= clean.year.max() else "outside fitted historical range (extrapolation)"
        st.success(f"Year {saved[4]}: {saved[5]:.8g} {units} — {cls}. This is a fitted estimate, not an actual observation.")
    st.subheader("Batch predictions")
    mode = st.radio("Batch input", ["Year range","Custom years"], horizontal=True)
    batch_years = []
    if mode == "Year range":
        b1,b2,b3 = st.columns(3)
        first = b1.number_input("Start year", 1950, 2100, max(1950,int(clean.year.max())+1), key="batch_start")
        last = b2.number_input("End year", 1950, 2100, min(2100,int(clean.year.max())+5), key="batch_end")
        step = b3.slider("Range step", 1, 10, 1)
        if first <= last:
            batch_years = list(range(int(first), int(last)+1, step))
        else:
            st.error("Start year must not exceed end year.")
    else:
        custom = st.text_input("Comma-separated years", "2025, 2030, 2040")
        for token in custom.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                val = int(token)
                if str(val) != token and not re.fullmatch(r"[+-]?\d+", token):
                    raise ValueError()
                if 1950 <= val <= 2100:
                    batch_years.append(val)
                else:
                    st.warning(f"Ignored {token}: years must be from 1950 through 2100.")
            except ValueError:
                st.warning(f"Ignored invalid year: {token}")
    batch_years = list(dict.fromkeys(batch_years))
    if batch_years:
        calc_df = pd.DataFrame({"year":batch_years})
        calc_df["estimate"] = model(calc_df.year.to_numpy()-base)
        calc_df["units"] = units
        calc_df["classification"] = np.where(calc_df.year.between(clean.year.min(),clean.year.max()),"within fitted range","outside fitted range (extrapolation)")
        st.dataframe(calc_df, use_container_width=True)
    # Comparison fetches
    compare_data = {}
    for cname in compare_names:
        try:
            compare_data[cname], _ = cached_fetch(COUNTRIES[cname], indicator_code, start_year, end_year)
        except Exception:
            compare_data[cname] = pd.DataFrame(columns=["year","value"])
    # Plot
    st.subheader("Historical data and fitted model")
    fig, ax = plt.subplots(figsize=(10,5))
    ax.scatter(clean.year, clean.value, label=f"{country_name} observed", s=24)
    smooth_years = np.linspace(clean.year.min(), clean.year.max()+horizon, 600)
    smooth_vals = model(smooth_years-base)
    hist_mask = smooth_years <= clean.year.max()
    ax.plot(smooth_years[hist_mask], smooth_vals[hist_mask], label="Fitted curve (historical)")
    if horizon > 0:
        ax.plot(smooth_years[~hist_mask], smooth_vals[~hist_mask], "--", label="Polynomial extrapolation")
        ax.axvline(clean.year.max(), linestyle=":", label="Historical boundary")
    marker_years = clean.year.iloc[::sample_step].to_numpy()
    ax.scatter(marker_years, model(marker_years-base), marker="x", label=f"Model markers every {sample_step} observed years")
    if not calc_df.empty:
        ax.scatter(calc_df.year, calc_df.estimate, marker="D", label="Batch estimates")
    for cname, df in compare_data.items():
        if not df.empty:
            ax.plot(df.year, df.value, marker=".", linewidth=1, label=f"{cname} observed")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"{indicator_name} ({units})")
    ax.set_title(f"{indicator_name}: {country_name}")
    ax.grid(True, alpha=.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    chart_bytes = buf.getvalue()
    plt.close(fig)
    st.caption("Polynomial extrapolations may become implausible. No prediction interval is implied. MAE describes typical in-sample absolute residual size, not a validated 95% interval.")

# Course and context lab
st.divider()
st.header("Interactive Function & Calculus Lab")
course = st.selectbox("Mathematics pathway", ["Honors Algebra 2","Precalculus","Calculus"])
context = st.selectbox("Context", list(CONTEXTS))
suggest_y, suggest_yu, suggest_x, suggest_xu, context_note = CONTEXTS[context]
default_fn = FUNCTION_PRESETS["Quadratic"]
if context == "Weather": default_fn = "a*sin(b*x) + c"
if context in ("Biology","Medical"): default_fn = "a*exp(b*x) + c"
if context == "Engineering": default_fn = "a*x**2 + b*x + c"
fn_kind = st.selectbox("Suggested function family", list(FUNCTION_PRESETS))
preset_fn = FUNCTION_PRESETS[fn_kind]
customize = st.checkbox("Customize context, variables, units, and description", value=False)
if customize:
    context_description = st.text_area("Context description", context_note)
    indep_name = st.text_input("Independent variable name", suggest_x)
    indep_units = st.text_input("Independent variable units", suggest_xu)
    dep_name = st.text_input("Dependent variable name", suggest_y)
    dep_units = st.text_input("Dependent variable units", suggest_yu)
else:
    context_description, indep_name, indep_units, dep_name, dep_units = context_note, suggest_x, suggest_xu, suggest_y, suggest_yu

st.caption(context_description)
st.write(f"Independent variable: **{indep_name}** ({indep_units}) · Dependent variable: **{dep_name}** ({dep_units})")
use_fit = st.checkbox("Use generated function from historical polynomial fit", value=False, disabled=not bool(fit_result))
params = {}
pcols = st.columns(4)
for i, pname in enumerate(["a","b","c","d"]):
    params[pname] = pcols[i].number_input(f"Parameter {pname}", value=[1.0,0.2,2.0,0.0][i], key=f"param_{pname}")
if use_fit and fit_result:
    # Convert fitted polynomial from centered year to a symbolic polynomial in x (year).
    base, coeff, _, _, clean = fit_result
    fn_expr = sum(sp.Float(c)*(x-base)**(len(coeff)-1-i) for i,c in enumerate(coeff))
    expression_text = f"Fitted data polynomial, x is calendar year (base {base})"
else:
    expression_text = st.text_input("Enter a function (use x, a, b, c, d; e.g. a*x**2+b*x+c)", preset_fn)
    try:
        fn_expr = safe_expression(expression_text, params)
    except Exception as exc:
        fn_expr = None
        st.error(f"Function input error: {exc}")

if fn_expr is not None:
    st.latex(sp.latex(sp.Eq(sp.Symbol(dep_name.replace(" ","_")), fn_expr)))
    interval_cols = st.columns(3)
    domain_min = interval_cols[0].number_input("Analysis interval start", value=0.0)
    domain_max = interval_cols[1].number_input("Analysis interval end", value=10.0)
    current_x = interval_cols[2].slider("Current input x", min_value=float(min(domain_min,domain_max)), max_value=float(max(domain_min,domain_max)), value=float(min(max(2.0,min(domain_min,domain_max)),max(domain_min,domain_max))))
    if domain_min >= domain_max:
        st.error("Analysis interval start must be less than interval end.")
    else:
        d1, d2 = sp.diff(fn_expr,x), sp.diff(fn_expr,x,2)
        value_now = numerical(fn_expr, np.array([current_x]))[0]
        slope_now = numerical(d1, np.array([current_x]))[0]
        accel_now = numerical(d2, np.array([current_x]))[0]
        if np.isfinite(value_now):
            st.metric(f"{dep_name} at x={current_x:g}", f"{value_now:.6g} {dep_units}")
        else:
            st.warning("Function is undefined at the selected input.")
        st.write(f"**First derivative:**")
        st.latex(sp.latex(d1))
        st.write(f"**Second derivative:**")
        st.latex(sp.latex(d2))
        if course == "Calculus":
            st.write(f"At x={current_x:g}, first derivative = {slope_now:.6g} {dep_units}/{indep_units}; second derivative = {accel_now:.6g} {dep_units}/{indep_units}².")
        elif course == "Precalculus":
            st.write("The first derivative describes local change; the second derivative describes how that rate changes. Use the graph to investigate turning behavior and concavity.")
        else:
            st.write("Explore domain, range, end behavior, graph trends, and average rates of change.")
        a1,a2 = st.columns(2)
        sec_a = a1.slider("Secant interval start", float(domain_min), float(domain_max), float(domain_min))
        sec_b = a2.slider("Secant interval end", float(domain_min), float(domain_max), float(domain_max))
        ya,yb = numerical(fn_expr,np.array([sec_a,sec_b]))
        if sec_a != sec_b and np.isfinite([ya,yb]).all():
            avg_rate = (yb-ya)/(sec_b-sec_a)
            st.write(f"Average rate of change from {sec_a:g} to {sec_b:g}: **{avg_rate:.6g} {dep_units}/{indep_units}**.")
        else:
            st.warning("Choose distinct endpoints where the function is defined.")
        show_f = st.checkbox("Show function", True)
        show_d1 = st.checkbox("Show first derivative", course=="Calculus")
        show_d2 = st.checkbox("Show second derivative", False)
        show_tangent = st.checkbox("Show tangent at current input", True)
        show_secant = st.checkbox("Show secant line", True)
        points = np.linspace(domain_min,domain_max,1200)
        yv = numerical(fn_expr,points)
        fig2, ax2 = plt.subplots(figsize=(10,5))
        if show_f: ax2.plot(points,np.where(np.isfinite(yv),yv,np.nan),label="Function")
        if show_d1:
            v=numerical(d1,points); ax2.plot(points,np.where(np.isfinite(v),v,np.nan),label="First derivative")
        if show_d2:
            v=numerical(d2,points); ax2.plot(points,np.where(np.isfinite(v),v,np.nan),label="Second derivative")
        if np.isfinite(value_now):
            ax2.scatter([current_x],[value_now],label="Current point",zorder=5)
            if show_tangent and np.isfinite(slope_now):
                tangent = value_now+slope_now*(points-current_x)
                ax2.plot(points,tangent,"--",label="Tangent line")
        if show_secant and sec_a != sec_b and np.isfinite([ya,yb]).all():
            sec_line=ya+(yb-ya)/(sec_b-sec_a)*(points-sec_a)
            ax2.plot(points,sec_line,":",label="Secant line")
        ax2.axhline(0,color="gray",linewidth=.7)
        ax2.set_xlabel(f"{indep_name} ({indep_units})")
        ax2.set_ylabel(f"{dep_name} / derivative values (check units)")
        ax2.grid(True,alpha=.25); ax2.legend(fontsize=8); fig2.tight_layout()
        st.pyplot(fig2); plt.close(fig2)
        if course == "Calculus":
            _,_,grid,vals,dy,ddy,critical,inflections = derivative_report(fn_expr,domain_min,domain_max)
            st.subheader("Calculus analysis (numerical candidates)")
            st.write(f"Critical-point candidates where f′ is approximately zero: {critical or 'none detected'}.")
            st.write(f"Inflection candidates with a detected concavity sign change: {inflections or 'none detected'}.")
            st.caption("Numerical grid search may miss closely spaced roots, even-multiplicity roots, or singularities. A zero of f′ is not automatically an extremum; a zero of f″ is not automatically an inflection point.")
            # Derivative sign-based local classification around candidates
            classifications=[]
            for p in critical:
                eps=max((domain_max-domain_min)/10000,1e-5)
                l,r=numerical(d1,np.array([p-eps,p+eps]))
                kind="local maximum" if l>0 and r<0 else ("local minimum" if l<0 and r>0 else "not classified as an extremum")
                classifications.append(f"x≈{p}: {kind} (first-derivative sign test)")
            st.write(" · ".join(classifications) if classifications else "No local extrema classified.")
            finite_vals=vals[np.isfinite(vals)]
            if finite_vals.size:
                st.write(f"Approximate range on selected interval (finite sampled values): [{finite_vals.min():.6g}, {finite_vals.max():.6g}].")
        else:
            st.write("**Domain and range:** inspect where the curve is defined and its displayed output values. **End behavior:** change the interval to explore larger positive and negative inputs.")
        # Average rates between adjacent points for precalculus
        if course in ("Honors Algebra 2","Precalculus"):
            st.write("**Changing average rates:** compare secant slopes over shorter and longer intervals to see how the rate changes.")

        # Simple context-specific animated indicator; Streamlit reruns on click.
        st.subheader("Visual simulation")
        visual = st.selectbox("Simulation visual", ["Walking/prosthetic-leg stride","Moving object/projectile","Tank or concentration gauge","Population/cell colony","Thermometer/weather","Business/economic indicator"])
        st.caption("The graphic is a conceptual indicator; its scale is normalized and does not represent a validated physical or medical simulation.")
        if np.isfinite(value_now):
            norm = 0.5 + 0.45*np.tanh(value_now/10)
            st.progress(float(np.clip(norm,0,1)), text=f"Normalized display for {visual}: {norm:.0%}")
        speed = st.select_slider("Playback speed (manual step)", options=["Slow","Normal","Fast"], value="Normal")
        if "play_index" not in st.session_state: st.session_state.play_index=0
        p1,p2,p3 = st.columns(3)
        if p1.button("▶ Play / next step"):
            st.session_state.play_index += {"Slow":1,"Normal":2,"Fast":4}[speed]
            st.session_state.sim_x = min(domain_max,domain_min+(st.session_state.play_index%100)/99*(domain_max-domain_min))
            st.rerun()
        if p2.button("Ⅱ Pause"):
            st.session_state.sim_x = current_x
            st.info("Playback paused. Move the current-input slider to continue exploring.")
        if p3.button("↺ Reset"):
            st.session_state.play_index=0
            st.session_state.sim_x=domain_min
            st.rerun()

# Exports
st.divider()
st.header("Export current analysis")
export_data = validated.copy() if "validated" in locals() else pd.DataFrame(columns=["year","value"])
csv_bytes = export_data.to_csv(index=False).encode("utf-8")
st.download_button("Download working data CSV", csv_bytes, file_name="historical_working_data.csv", mime="text/csv")
summary = pd.DataFrame([{"Country":country_name,"Indicator":indicator_name,"Indicator code":indicator_code,"Units":units,"Degree":degree,"Equation":equation,"R2 in-sample":r2,"MAE in-sample":mae,"User modified":st.session_state.get("modified",False),"Retrieved":datetime.now().isoformat(timespec="seconds")}])
xlsx_buf=io.BytesIO()
with pd.ExcelWriter(xlsx_buf,engine="xlsxwriter") as writer:
    summary.to_excel(writer,sheet_name="Summary",index=False)
    export_data.to_excel(writer,sheet_name="Working Data",index=False)
    calc_df.to_excel(writer,sheet_name="Calculated Years",index=False)
st.download_button("Download Excel workbook",xlsx_buf.getvalue(),file_name="historical_data_explorer.xlsx",mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
details=[f"Country: {country_name}",f"Indicator: {indicator_name} ({indicator_code})",f"Definition: {definition}",f"Units: {units}",f"Retrieval date: {datetime.now().date()}",f"User-modified scenario: {st.session_state.get('modified',False)}",f"Equation: {equation}",f"In-sample R²: {r2}",f"In-sample MAE: {mae}", "Limitations: polynomial extrapolation is uncertain; no 95% prediction interval is claimed."]
pdf_bytes=make_pdf("Hispanic Heritage Historical Data Explorer",details,chart_bytes,export_data,calc_df)
st.download_button("Download PDF report",pdf_bytes,file_name="historical_data_explorer_report.pdf",mime="application/pdf")

if printer_mode:
    st.markdown("---")
    st.subheader("Printer-friendly summary")
    st.write(f"{country_name} · {indicator_name} · {units} · observations: {len(export_data)} · degree: {degree}")
    st.write(equation)
    st.write(f"In-sample R²: {r2:.5f} · In-sample MAE: {mae:.5g}" if np.isfinite(r2) and np.isfinite(mae) else "Regression not available.")
