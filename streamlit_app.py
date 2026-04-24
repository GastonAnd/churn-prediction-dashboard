"""Dashboard interactivo de Customer Churn con Streamlit.

Este frontend consume la API real de FastAPI para:
- EDA sobre datos reales del dataset
- Prediccion individual con modelos de ML entrenados en API
"""

from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
import plotly.express as px
import streamlit as st


def configure_page() -> None:
    st.set_page_config(
        page_title="Telecom Churn Intelligence",
        page_icon="📉",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }
            .hero {
                background: linear-gradient(120deg, #0d1b2a 0%, #1b263b 45%, #415a77 100%);
                color: #f8f9fa;
                border-radius: 14px;
                padding: 1.3rem 1.5rem;
                margin-bottom: 1rem;
                box-shadow: 0 10px 30px rgba(13, 27, 42, 0.25);
            }
            .hero p {
                margin-bottom: 0;
                opacity: 0.95;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def call_api_json(url: str, method: str = "GET", payload: dict | None = None) -> dict:
    headers = {"Content-Type": "application/json"}
    data_bytes = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url=url, data=data_bytes, headers=headers, method=method)
    with urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


@st.cache_data(ttl=30)
def fetch_health(api_base_url: str) -> dict:
    return call_api_json(f"{api_base_url}/health")


@st.cache_data(ttl=30)
def fetch_predictions_dataframe(api_base_url: str, model_name: str, limit: int) -> pd.DataFrame:
    payload = call_api_json(f"{api_base_url}/predictions?model={model_name}&limit={limit}")
    return pd.DataFrame(payload.get("rows", []))


@st.cache_data(ttl=60)
def fetch_eda_images(api_base_url: str) -> list[str]:
    payload = call_api_json(f"{api_base_url}/eda-images")
    base = api_base_url.rstrip("/")
    return [f"{base}{path}" for path in payload.get("images", [])]


def render_header() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1 style="margin:0;">Telecom Churn Intelligence Dashboard</h1>
            <p>
                Proyecto de Machine Learning para estimar la probabilidad de fuga de clientes
                y explorar patrones de comportamiento en una telco.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> tuple[str, str, str]:
    st.sidebar.title("Navegacion")
    api_url = st.sidebar.text_input("URL de API", value="http://127.0.0.1:8000")
    eda_model = st.sidebar.selectbox("Modelo para EDA", ("tree", "logistic"), index=0)
    selected_tab = st.sidebar.radio(
        "Selecciona una seccion:",
        ("Analisis Exploratorio (EDA)", "Simulador Predictivo"),
    )
    return selected_tab, api_url.rstrip("/"), eda_model


def render_eda(df: pd.DataFrame, image_urls: list[str], eda_model: str) -> None:
    st.subheader("Analisis Exploratorio (EDA)")
    st.write("Analisis basado en datos reales servidos por la API.")

    if df.empty:
        st.warning("La API no devolvio filas para el analisis EDA.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Clientes", f"{len(df):,}")
    col2.metric("Tasa de churn", f"{df['Churn'].mean() * 100:.1f}%")
    col3.metric("Gasto mensual promedio", f"${df['MonthlyCharge'].mean():.2f}")
    st.caption(f"Modelo seleccionado para obtener datos enriquecidos: {eda_model}")

    churn_labels = df["Churn"].map({0: "No", 1: "Si"})

    fig_churn = px.histogram(
        df.assign(ChurnLabel=churn_labels),
        x="ChurnLabel",
        color="ChurnLabel",
        text_auto=True,
        title="Distribucion de Churn",
        category_orders={"ChurnLabel": ["No", "Si"]},
        color_discrete_map={"No": "#2a9d8f", "Si": "#e76f51"},
    )
    fig_churn.update_layout(showlegend=False, xaxis_title="Churn", yaxis_title="Clientes")
    st.plotly_chart(fig_churn, use_container_width=True)

    fig_box = px.box(
        df.assign(ChurnLabel=churn_labels),
        x="ChurnLabel",
        y="MonthlyCharge",
        color="ChurnLabel",
        title="MonthlyCharge vs Churn",
        category_orders={"ChurnLabel": ["No", "Si"]},
        color_discrete_map={"No": "#457b9d", "Si": "#d62828"},
    )
    fig_box.update_layout(xaxis_title="Churn", yaxis_title="Monthly Charge")
    st.plotly_chart(fig_box, use_container_width=True)

    fig_scatter = px.scatter(
        df.assign(ChurnLabel=churn_labels),
        x="CustServCalls",
        y="MonthlyCharge",
        color="ChurnLabel",
        size="DataUsage",
        hover_data=["AccountWeeks", "OverageFee", "RoamMins"],
        title="Relacion entre soporte, gasto mensual y churn",
        color_discrete_map={"No": "#3a86ff", "Si": "#ff006e"},
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("### Graficas EDA generadas por backend")
    if image_urls:
        for image_url in image_urls[:8]:
            st.image(image_url, use_container_width=True)
    else:
        st.info("No se encontraron imagenes EDA en la API.")

    st.dataframe(df.head(20), use_container_width=True)


def render_predictive_simulator(api_base_url: str) -> None:
    st.subheader("Simulador Predictivo")
    st.write("Ajusta variables y ejecuta prediccion real mediante la API.")

    with st.form("simulator_form"):
        col1, col2 = st.columns(2)

        with col1:
            contract_renewal = st.selectbox("ContractRenewal", ("Si", "No"), index=0)
            data_plan = st.selectbox("DataPlan", ("Si", "No"), index=0)
            account_weeks = st.slider("AccountWeeks", min_value=1, max_value=260, value=110)
            day_mins = st.slider("DayMins", min_value=0.0, max_value=360.0, value=180.0, step=0.5)
            day_calls = st.slider("DayCalls", min_value=0, max_value=200, value=100)
            monthly_charge = st.slider(
                "MonthlyCharge", min_value=15.0, max_value=150.0, value=67.5, step=0.5
            )

        with col2:
            cust_serv_calls = st.slider("CustServCalls", min_value=0, max_value=12, value=2)
            data_usage = st.slider("DataUsage", min_value=0.0, max_value=8.0, value=2.1, step=0.1)
            overage_fee = st.slider("OverageFee", min_value=0.0, max_value=30.0, value=9.5, step=0.1)
            roam_mins = st.slider("RoamMins", min_value=0.0, max_value=50.0, value=10.0, step=0.5)

        submitted = st.form_submit_button("Calcular riesgo")

    if not submitted:
        st.info("Completa el formulario y presiona 'Calcular riesgo'.")
        return

    client_df = pd.DataFrame(
        [
            {
                "ContractRenewal": 1 if contract_renewal == "Si" else 0,
                "DataPlan": 1 if data_plan == "Si" else 0,
                "AccountWeeks": account_weeks,
                "DayMins": day_mins,
                "DayCalls": day_calls,
                "MonthlyCharge": monthly_charge,
                "CustServCalls": cust_serv_calls,
                "DataUsage": data_usage,
                "OverageFee": overage_fee,
                "RoamMins": roam_mins,
            }
        ]
    )

    try:
        result = call_api_json(
            f"{api_base_url}/predict", method="POST", payload=client_df.iloc[0].to_dict()
        )
    except (HTTPError, URLError, TimeoutError, ValueError) as exc:
        st.error(f"No fue posible consultar la API para predecir: {exc}")
        return

    logistic_prob = float(result["logistic"]["churn_probability"])
    tree_prob = float(result["tree_entropy"]["churn_probability"])
    final_prob = max(logistic_prob, tree_prob)
    risk_level = "ALTO" if final_prob >= 0.5 else "BAJO"

    c1, c2 = st.columns(2)
    c1.metric("Probabilidad de Fuga (Logistic)", f"{logistic_prob * 100:.1f}%")
    c2.metric("Probabilidad de Fuga (Tree Entropy)", f"{tree_prob * 100:.1f}%")

    if final_prob >= 0.5:
        st.error(f"Riesgo {risk_level}: conviene activar una estrategia de retencion.")
    else:
        st.success(f"Riesgo {risk_level}: cliente con buena salud de permanencia.")


def main() -> None:
    configure_page()
    inject_styles()
    render_header()

    selected_tab, api_base_url, eda_model = render_sidebar()

    try:
        health = fetch_health(api_base_url)
        st.caption(
            f"API conectada: {api_base_url} | filas={health.get('rows', '?')} | columnas={health.get('columns', '?')}"
        )
    except (HTTPError, URLError, TimeoutError, ValueError) as exc:
        st.error(f"No se pudo conectar con la API en {api_base_url}: {exc}")
        st.info("Levanta la API con: python -m uvicorn api.main:app --reload")
        return

    if selected_tab == "Analisis Exploratorio (EDA)":
        df = fetch_predictions_dataframe(api_base_url, model_name=eda_model, limit=3333)
        image_urls = fetch_eda_images(api_base_url)
        render_eda(df, image_urls, eda_model)
    else:
        render_predictive_simulator(api_base_url)


if __name__ == "__main__":
    main()
