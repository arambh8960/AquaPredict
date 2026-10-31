
"""
AquaPredict - Water Quality Prediction Streamlit Dashboard
Water Quality Analysis + Spark MLlib + BIS-Based Screening
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


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AquaPredict | Water Quality Analysis",
    page_icon="💧",
    layout="wide"
)


# =========================================================
# CACHED FUNCTIONS
# =========================================================

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

    return df.fillna(df.mean(numeric_only=True))


# Load lightweight dashboard resources
metrics_data = load_metrics()
df = load_raw_data()


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

st.sidebar.image(
    "https://img.icons8.com/fluency/96/water.png",
    width=80
)

st.sidebar.title("AquaPredict")

st.sidebar.caption(
    "Water Quality Analysis using Apache Spark MLlib"
)

page = st.sidebar.radio(
    "Navigate Pages",
    [
        "🏠 Home Dashboard",
        "🚰 Water Prediction",
        "📊 Model Comparison",
        "📈 Data Analytics"
    ]
)


# =========================================================
# PAGE 1: HOME DASHBOARD
# =========================================================

if page == "🏠 Home Dashboard":

    st.title("💧 AquaPredict: Water Quality Analysis")

    st.markdown(
        "### Powered by Apache Spark, Spark MLlib & Streamlit"
    )

    st.write(
        "An interactive water quality analysis platform "
        "that uses machine learning to predict water potability "
        "based on nine physicochemical parameters."
    )

    st.info(
        "AquaPredict provides preliminary water quality screening. "
        "It is not a certified drinking-water safety test."
    )

    st.markdown("---")

    total_records = len(df)

    potable_count = int(df["Potability"].sum())

    non_potable_count = total_records - potable_count

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Water Samples",
        f"{total_records:,}"
    )

    col2.metric(
        "Potable Samples",
        f"{potable_count:,}"
    )

    col3.metric(
        "Non-Potable Samples",
        f"{non_potable_count:,}"
    )

    col4.metric(
        "Spark Master",
        "local[*]"
    )

    st.markdown("---")

    c1, c2 = st.columns([1.5, 1])

    with c1:

        st.subheader("Dataset Summary Statistics")

        st.dataframe(
            df.describe().T.style.format("{:.2f}"),
            use_container_width=True
        )

    with c2:

        st.subheader("Potability Class Balance")

        fig = px.pie(
            names=[
                "Non-Potable (0)",
                "Potable (1)"
            ],
            values=[
                non_potable_count,
                potable_count
            ],
            color_discrete_sequence=[
                "#EF553B",
                "#00CC96"
            ],
            hole=0.45
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# =========================================================
# PAGE 2: WATER QUALITY ANALYSIS AND PREDICTION
# =========================================================

elif page == "🚰 Water Prediction":

    st.title("🚰 Water Quality Analysis & Prediction")

    st.write(
        "Enter the water quality parameters to analyze "
        "the sample and predict its potability."
    )

    st.info(
        "Enter actual laboratory-measured values whenever possible. "
        "Default values are only sample inputs."
    )

    selected_model = st.selectbox(
        "Choose Spark MLlib Algorithm:",
        [
            "Random Forest",
            "Decision Tree",
            "Logistic Regression",
            "Naive Bayes"
        ]
    )

    st.markdown("---")

    # -----------------------------------------------------
    # USER INPUTS
    # -----------------------------------------------------

    st.subheader("🧪 Enter Water Quality Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:

        ph = st.number_input(
            "pH:",
            min_value=0.0,
            max_value=14.0,
            value=7.2,
            step=0.1
        )

        hardness = st.number_input(
            "Hardness (mg/L):",
            min_value=0.0,
            max_value=10000.0,
            value=204.0,
            step=1.0
        )

        solids = st.number_input(
            "Total Dissolved Solids (mg/L):",
            min_value=0.0,
            max_value=100000.0,
            value=20790.0,
            step=100.0
        )

    with col2:

        chloramines = st.number_input(
            "Chloramines (mg/L):",
            min_value=0.0,
            max_value=100.0,
            value=7.3,
            step=0.1
        )

        sulfate = st.number_input(
            "Sulfate (mg/L):",
            min_value=0.0,
            max_value=10000.0,
            value=368.0,
            step=1.0
        )

        conductivity = st.number_input(
            "Conductivity (µS/cm):",
            min_value=0.0,
            max_value=100000.0,
            value=564.0,
            step=5.0
        )

    with col3:

        organic_carbon = st.number_input(
            "Organic Carbon (mg/L):",
            min_value=0.0,
            max_value=1000.0,
            value=10.4,
            step=0.1
        )

        trihalomethanes = st.number_input(
            "Trihalomethanes (µg/L):",
            min_value=0.0,
            max_value=10000.0,
            value=87.0,
            step=0.5
        )

        turbidity = st.number_input(
            "Turbidity (NTU):",
            min_value=0.0,
            max_value=1000.0,
            value=3.0,
            step=0.1
        )

    # -----------------------------------------------------
    # INPUT DATA
    # -----------------------------------------------------

    input_sample = {
        "ph": ph,
        "Hardness": hardness,
        "Solids": solids,
        "Chloramines": chloramines,
        "Sulfate": sulfate,
        "Conductivity": conductivity,
        "Organic_carbon": organic_carbon,
        "Trihalomethanes": trihalomethanes,
        "Turbidity": turbidity
    }

    st.markdown("---")

    # -----------------------------------------------------
    # ANALYSIS BUTTON
    # -----------------------------------------------------

    if st.button(
        "🔍 Analyze Water Quality",
        use_container_width=True,
        type="primary"
    ):

        # ================================================
        # STEP 1: BIS-BASED PARAMETER SCREENING
        # ================================================

        st.subheader("📊 Water Quality Parameter Analysis")

        checks = [

            {
                "Parameter": "pH",
                "Value": ph,
                "Unit": "",
                "Acceptable Limit": "6.5 - 8.5",
                "Permissible Limit": "No relaxation",
                "passed": 6.5 <= ph <= 8.5,
                "permissible": 6.5 <= ph <= 8.5
            },

            {
                "Parameter": "Total Dissolved Solids",
                "Value": solids,
                "Unit": "mg/L",
                "Acceptable Limit": "500",
                "Permissible Limit": "2000",
                "passed": solids <= 500,
                "permissible": solids <= 2000
            },

            {
                "Parameter": "Total Hardness",
                "Value": hardness,
                "Unit": "mg/L",
                "Acceptable Limit": "200",
                "Permissible Limit": "600",
                "passed": hardness <= 200,
                "permissible": hardness <= 600
            },

            {
                "Parameter": "Sulfate",
                "Value": sulfate,
                "Unit": "mg/L",
                "Acceptable Limit": "200",
                "Permissible Limit": "400",
                "passed": sulfate <= 200,
                "permissible": sulfate <= 400
            },

            {
                "Parameter": "Turbidity",
                "Value": turbidity,
                "Unit": "NTU",
                "Acceptable Limit": "1",
                "Permissible Limit": "5",
                "passed": turbidity <= 1,
                "permissible": turbidity <= 5
            }
        ]

        result_rows = []

        for item in checks:

            if item["passed"]:

                status = "✅ Acceptable"

            elif item["permissible"]:

                status = "⚠️ Permissible Range"

            else:

                status = "❌ Exceeds Permissible Limit"

            result_rows.append({
                "Parameter": item["Parameter"],
                "Measured Value": item["Value"],
                "Unit": item["Unit"],
                "Acceptable Limit": item["Acceptable Limit"],
                "Permissible Limit": item["Permissible Limit"],
                "Status": status
            })

        checks_df = pd.DataFrame(result_rows)

        st.dataframe(
            checks_df,
            use_container_width=True,
            hide_index=True
        )

        # ================================================
        # STEP 2: SUMMARY METRICS
        # ================================================

        acceptable_count = sum(
            item["passed"] for item in checks
        )

        permissible_count = sum(
            (
                not item["passed"]
                and item["permissible"]
            )
            for item in checks
        )

        exceeded_count = sum(
            not item["permissible"]
            for item in checks
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Acceptable Parameters",
            acceptable_count
        )

        col2.metric(
            "Permissible Range",
            permissible_count
        )

        col3.metric(
            "Exceeding Limits",
            exceeded_count
        )

        if exceeded_count > 0:

            st.error(
                "One or more tested parameters exceed "
                "the displayed permissible limits."
            )

        elif permissible_count > 0:

            st.warning(
                "Some parameters are within the permissible "
                "range but exceed their acceptable limits."
            )

        else:

            st.success(
                "All five screened parameters are within "
                "their acceptable limits."
            )

        # ================================================
        # STEP 3: MODEL INPUT RANGE WARNING
        # ================================================

        st.markdown("---")

        st.subheader("🔎 Model Input Validation")

        dataset_columns = {
            "ph": "ph",
            "Hardness": "Hardness",
            "Solids": "Solids",
            "Chloramines": "Chloramines",
            "Sulfate": "Sulfate",
            "Conductivity": "Conductivity",
            "Organic_carbon": "Organic_carbon",
            "Trihalomethanes": "Trihalomethanes",
            "Turbidity": "Turbidity"
        }

        out_of_range = []

        for input_key, dataset_key in dataset_columns.items():

            min_value = df[dataset_key].min()
            max_value = df[dataset_key].max()

            entered_value = input_sample[input_key]

            if (
                entered_value < min_value
                or entered_value > max_value
            ):

                out_of_range.append(
                    f"{dataset_key}: {min_value:.2f} to {max_value:.2f}"
                )

        if out_of_range:

            st.warning(
                "Some entered values are outside the training "
                "dataset range. Model predictions may be less reliable."
            )

            for item in out_of_range:
                st.write(f"- {item}")

        else:

            st.success(
                "All entered values fall within the training dataset ranges."
            )

        # ================================================
        # STEP 4: SPARK MLlib PREDICTION
        # ================================================

        st.markdown("---")

        st.subheader("🤖 Machine Learning Prediction")

        try:

            with st.spinner(
                "Loading Spark MLlib and analyzing water sample..."
            ):

                predictor, spark = load_predictor()

                result = predictor.predict(
                    input_sample,
                    selected_model
                )

            if result["prediction"] == 1:

                st.success(
                    f"### 💧 ML Prediction: {result['label']}"
                )

            else:

                st.error(
                    f"### ⚠️ ML Prediction: {result['label']}"
                )

            if result["probability_potable"] is not None:

                probability = result["probability_potable"]

                st.write(
                    "**Model-estimated probability of Potable class:** "
                    f"{probability * 100:.2f}%"
                )

                st.progress(
                    max(0.0, min(1.0, float(probability)))
                )

                st.caption(
                    "This value represents the model's estimated "
                    "class probability. It is NOT the purity percentage "
                    "or a laboratory measurement of water safety."
                )

        except Exception as e:

            st.error(
                f"Prediction failed: {str(e)}"
            )

            st.info(
                "Please check the Spark session, trained model files "
                "and application logs."
            )

        # ================================================
        # STEP 5: FINAL SAFETY INFORMATION
        # ================================================

        st.markdown("---")

        st.subheader("⚠️ Important Water Safety Information")

        st.warning(
            """
            AquaPredict is a preliminary screening and machine-learning
            prediction tool. It cannot certify water as pure or safe to drink.

            The parameter screening above checks only five available
            parameters against the displayed drinking-water limits.

            Other parameters such as chloramines, conductivity,
            organic carbon and trihalomethanes are not independently
            certified by this screening.

            Microbiological contaminants such as E. coli, bacteria
            and viruses are not measured by this dataset.

            A certified laboratory test is required before declaring
            water safe for drinking.
            """
        )


# =========================================================
# PAGE 3: MODEL COMPARISON
# =========================================================

elif page == "📊 Model Comparison":

    st.title("📊 Spark MLlib Algorithms Comparison")

    st.write(
        "Performance evaluation of four machine learning algorithms."
    )

    rows = []

    for model_name, m in metrics_data.items():

        rows.append({
            "Algorithm": model_name,
            "Accuracy (%)": round(m["accuracy"] * 100, 2),
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1-Score": m["f1_score"]
        })

    metrics_df = pd.DataFrame(rows).sort_values(
        by="Accuracy (%)",
        ascending=False
    )

    st.dataframe(
        metrics_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:

        st.subheader("Accuracy Comparison")

        fig = px.bar(
            metrics_df,
            x="Algorithm",
            y="Accuracy (%)",
            color="Algorithm",
            text="Accuracy (%)",
            color_discrete_sequence=px.colors.qualitative.Vivid
        )

        fig.update_layout(
            yaxis_range=[0, 100]
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with c2:

        st.subheader("F1-Score Comparison")

        fig = px.bar(
            metrics_df,
            x="Algorithm",
            y="F1-Score",
            color="Algorithm",
            text="F1-Score",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )

        fig.update_layout(
            yaxis_range=[0, 1]
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")

    st.subheader("Confusion Matrices")

    cm_cols = st.columns(4)

    for idx, (m_name, m) in enumerate(metrics_data.items()):

        cm = m["confusion_matrix"]

        fig_cm = px.imshow(
            cm,
            text_auto=True,
            labels={
                "x": "Predicted",
                "y": "Actual",
                "color": "Count"
            },
            x=[
                "Non-Potable (0)",
                "Potable (1)"
            ],
            y=[
                "Non-Potable (0)",
                "Potable (1)"
            ],
            title=m_name,
            color_continuous_scale="Blues"
        )

        cm_cols[idx % 4].plotly_chart(
            fig_cm,
            use_container_width=True
        )


# =========================================================
# PAGE 4: DATA ANALYTICS
# =========================================================

elif page == "📈 Data Analytics":

    st.title("📈 Exploratory Data Analytics")

    st.write(
        "Explore water quality features, distributions "
        "and relationships in the dataset."
    )

    st.subheader("Correlation Heatmap")

    corr = df.corr()

    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Pearson Correlation Heatmap of Water Parameters"
    )

    st.plotly_chart(
        fig_corr,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Feature Distributions by Potability")

    feature_to_plot = st.selectbox(
        "Select Feature:",
        [
            c for c in df.columns
            if c != "Potability"
        ]
    )

    fig_hist = px.histogram(
        df,
        x=feature_to_plot,
        color="Potability",
        barmode="overlay",
        marginal="box",
        color_discrete_map={
            0: "#EF553B",
            1: "#00CC96"
        },
        title=(
            f"Distribution of {feature_to_plot} "
            "grouped by Potability"
        )
    )

    st.plotly_chart(
        fig_hist,
        use_container_width=True
    )
