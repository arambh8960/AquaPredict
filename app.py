"""
AquaPredict - Water Quality Prediction Streamlit Dashboard
Multi-page Interactive Application powered by Apache Spark MLlib
Path: app.py
"""
import sys
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st
import pandas as pd
import plotly.express as px

from src.spark_session import get_spark_session
from src.predict import WaterPredictor

st.set_page_config(
    page_title="AquaPredict | Water Quality Prediction",
    page_icon="💧",
    layout="wide"
)

@st.cache_resource
def load_predictor():
    spark = get_spark_session("AquaPredict-Streamlit")
    return WaterPredictor(spark), spark

@st.cache_data
def load_metrics():
    metrics_path = BASE_DIR / "models" / "metrics.json"
    with open(metrics_path, "r") as f:
        return json.load(f)

@st.cache_data
def load_raw_data():
    csv_path = BASE_DIR / "data" / "water_potability.csv"
    df = pd.read_csv(csv_path)
    return df.fillna(df.mean())

predictor, spark = load_predictor()
metrics_data = load_metrics()
df = load_raw_data()

st.sidebar.image("https://img.icons8.com/fluency/96/water.png", width=80)
st.sidebar.title("AquaPredict")
st.sidebar.caption("Distributed Water Quality Prediction via Spark MLlib")
page = st.sidebar.radio("Navigate Pages", ["🏠 Home Dashboard", "🚰 Water Prediction", "📊 Model Comparison", "📈 Data Analytics"])

# PAGE 1: HOME DASHBOARD
if page == "🏠 Home Dashboard":
    st.title("💧 AquaPredict: Water Quality Prediction")
    st.markdown("### Powered by Apache Spark, Spark MLlib & Streamlit")
    st.write("Cluster computing project classifying water samples as Potable (1) or Non-Potable (0) using 9 biochemical parameters.")
    st.markdown("---")
    
    total_records = len(df)
    potable_count = int(df["Potability"].sum())
    non_potable_count = total_records - potable_count
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Water Samples", f"{total_records:,}")
    col2.metric("Potable Samples (Class 1)", f"{potable_count:,}", f"{(potable_count/total_records)*100:.1f}%")
    col3.metric("Non-Potable Samples (Class 0)", f"{non_potable_count:,}", f"-{(non_potable_count/total_records)*100:.1f}%")
    col4.metric("Spark Master", "local[*]")

    st.markdown("---")
    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.subheader("Dataset Summary Statistics")
        st.dataframe(df.describe().T.style.format("{:.2f}"), use_container_width=True)
    with c2:
        st.subheader("Potability Class Balance")
        fig = px.pie(
            names=["Non-Potable (0)", "Potable (1)"],
            values=[non_potable_count, potable_count],
            color_discrete_sequence=["#EF553B", "#00CC96"],
            hole=0.45
        )
        st.plotly_chart(fig, use_container_width=True)

# PAGE 2: WATER PREDICTION
elif page == "🚰 Water Prediction":
    st.title("🚰 Single Sample Water Potability Prediction")
    st.write("Enter parameter values below to predict potability:")

    selected_model = st.selectbox(
        "Choose Spark MLlib Algorithm:",
        ["Random Forest", "Decision Tree", "Logistic Regression", "Naive Bayes"]
    )
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        ph = st.number_input("pH (0 - 14):", min_value=0.0, max_value=14.0, value=7.2, step=0.1)
        hardness = st.number_input("Hardness (mg/L):", min_value=40.0, max_value=350.0, value=204.0, step=1.0)
        solids = st.number_input("Total Dissolved Solids (ppm):", min_value=300.0, max_value=65000.0, value=20790.0, step=100.0)

    with col2:
        chloramines = st.number_input("Chloramines (ppm):", min_value=0.0, max_value=15.0, value=7.3, step=0.1)
        sulfate = st.number_input("Sulfate (mg/L):", min_value=100.0, max_value=500.0, value=368.0, step=1.0)
        conductivity = st.number_input("Conductivity (μS/cm):", min_value=100.0, max_value=800.0, value=564.0, step=5.0)

    with col3:
        organic_carbon = st.number_input("Organic Carbon (ppm):", min_value=2.0, max_value=30.0, value=10.4, step=0.1)
        trihalomethanes = st.number_input("Trihalomethanes (μg/L):", min_value=0.0, max_value=130.0, value=87.0, step=0.5)
        turbidity = st.number_input("Turbidity (NTU):", min_value=1.0, max_value=7.0, value=3.0, step=0.1)

    input_sample = {
        "ph": ph, "Hardness": hardness, "Solids": solids,
        "Chloramines": chloramines, "Sulfate": sulfate,
        "Conductivity": conductivity, "Organic_carbon": organic_carbon,
        "Trihalomethanes": trihalomethanes, "Turbidity": turbidity
    }

    if st.button("🔍 Predict Water Potability", use_container_width=True, type="primary"):
        with st.spinner(f"Evaluating sample using Spark MLlib {selected_model}..."):
            result = predictor.predict(input_sample, selected_model)
            st.markdown("---")
            if result["prediction"] == 1:
                st.success(f"### 🎉 Result: {result['label']}")
            else:
                st.error(f"### ⚠️ Result: {result['label']}")

            if result["probability_potable"] is not None:
                prob = result["probability_potable"]
                st.write(f"**Confidence Probability of Potability:** `{prob*100:.2f}%`")
                st.progress(prob)

# PAGE 3: MODEL COMPARISON
elif page == "📊 Model Comparison":
    st.title("📊 Spark MLlib Algorithms Comparison")
    st.write("Performance evaluation of 4 algorithms tested on 601 records.")

    rows = []
    for model_name, m in metrics_data.items():
        rows.append({
            "Algorithm": model_name,
            "Accuracy (%)": round(m["accuracy"] * 100, 2),
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1-Score": m["f1_score"]
        })
    metrics_df = pd.DataFrame(rows).sort_values(by="Accuracy (%)", ascending=False)
    st.table(metrics_df)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Accuracy Comparison")
        fig = px.bar(
            metrics_df, x="Algorithm", y="Accuracy (%)",
            color="Algorithm", text="Accuracy (%)",
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
        fig.update_layout(yaxis_range=[50, 75])
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("F1-Score Comparison")
        fig = px.bar(
            metrics_df, x="Algorithm", y="F1-Score",
            color="Algorithm", text="F1-Score",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig.update_layout(yaxis_range=[0.3, 0.7])
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Confusion Matrices (Test Set: 601 Records)")
    cm_cols = st.columns(4)
    for idx, (m_name, m) in enumerate(metrics_data.items()):
        cm = m["confusion_matrix"]
        fig_cm = px.imshow(
            cm, text_auto=True,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Non-Potable (0)", "Potable (1)"],
            y=["Non-Potable (0)", "Potable (1)"],
            title=m_name,
            color_continuous_scale="Blues"
        )
        cm_cols[idx].plotly_chart(fig_cm, use_container_width=True)

# PAGE 4: DATA ANALYTICS
elif page == "📈 Data Analytics":
    st.title("📈 Exploratory Data Analytics (EDA)")
    st.write("Feature distributions, correlation matrix, and biochemical relationships.")

    st.subheader("Correlation Heatmap")
    corr = df.corr()
    fig_corr = px.imshow(
        corr, text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Pearson Correlation Heatmap of Water Parameters"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("---")
    st.subheader("Feature Distributions by Potability")
    feature_to_plot = st.selectbox("Select Feature:", [c for c in df.columns if c != "Potability"])
    
    fig_hist = px.histogram(
        df, x=feature_to_plot, color="Potability",
        barmode="overlay", marginal="box",
        color_discrete_map={0: "#EF553B", 1: "#00CC96"},
        title=f"Distribution of {feature_to_plot} grouped by Potability"
    )
    st.plotly_chart(fig_hist, use_container_width=True)
