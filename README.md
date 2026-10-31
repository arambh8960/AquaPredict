 <div align="center">

# 💧 AquaPredict

### Water Quality Prediction Using Apache Spark and Machine Learning

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.5.1-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white)
![PySpark](https://img.shields.io/badge/PySpark-MLlib-FDEE21?style=for-the-badge&logo=apachespark&logoColor=black)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Data-Pandas-150458?style=for-the-badge&logo=pandas)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

**AquaPredict** is a machine learning-based water quality prediction platform that uses **Apache Spark, PySpark MLlib, and Streamlit** to analyze water quality parameters and predict whether a water sample is potable or non-potable.

The platform combines data preprocessing, distributed data processing, machine learning classification, and interactive visualization to make water quality analysis easier and more accessible.

**Built to demonstrate the application of Big Data Processing and Machine Learning in water quality analysis.**

🌐 **Live Demo:** https://aquapredict8960.streamlit.app/

📂 **GitHub Repository:** https://github.com/arambh8960/AquaPredict

</div>

---

# ✨ Why AquaPredict?

Access to safe drinking water is essential for public health. Evaluating water quality involves analyzing multiple physical and chemical parameters.

AquaPredict simplifies this process by using machine learning models to predict water potability based on water quality measurements.

Instead of manually interpreting multiple parameters, users can enter water quality values and receive a model-based prediction.

---

# 🚀 Features

## 💧 Water Potability Prediction

- Predict whether water is potable or non-potable.
- Enter water quality parameters through an interactive interface.
- Generate predictions using trained machine learning models.
- View prediction results through the Streamlit application.

## ⚡ Apache Spark Integration

- Process water quality data using PySpark.
- Initialize Spark sessions for data processing.
- Perform data operations using Spark DataFrames.
- Demonstrate RDD operations and Spark SQL.
- Use Spark MLlib for machine learning.

## 🤖 Machine Learning Models

AquaPredict implements four classification algorithms:

- Logistic Regression
- Decision Tree Classifier
- Naive Bayes
- Random Forest Classifier

## 📊 Model Comparison

- Compare the performance of multiple ML algorithms.
- Display model evaluation metrics.
- Analyze classification performance.
- Identify differences between trained models.

## 📈 Data Analytics and Visualization

- Explore water quality parameters.
- Analyze dataset distributions.
- Visualize important data patterns.
- Present analytical insights through interactive charts.

## 🖥️ Interactive Web Application

- Built using Streamlit.
- User-friendly navigation.
- Interactive prediction interface.
- Model comparison dashboard.
- Data analytics dashboard.

---

# 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Programming Language | Python 3.11 |
| Big Data Processing | Apache Spark 3.5.1 |
| Spark API | PySpark |
| Machine Learning | Spark MLlib |
| Data Manipulation | Pandas, NumPy |
| Visualization | Plotly, Matplotlib, Seaborn |
| Web Framework | Streamlit |
| Development Tools | Git, GitHub, VS Code |
| Deployment | Streamlit Community Cloud |

---

# 🏗️ System Architecture

```text
                 User
                   │
                   ▼
          Streamlit Web Application
                   │
          ┌────────┴────────┐
          │                 │
          ▼                 ▼
    User Input         Data Analytics
          │                 │
          ▼                 ▼
     PySpark Session    Pandas / Plotly
          │
          ▼
     Data Preprocessing
          │
          ▼
      Feature Assembly
          │
          ▼
       Feature Scaling
          │
          ▼
       Spark MLlib
          │
     ┌────┼───────────────┐
     │    │               │
     ▼    ▼               ▼
 Logistic Decision     Random
 Regression Tree       Forest
     │    │               │
     └────┴──────┬────────┘
                 │
                 ▼
          Prediction Result
                 │
                 ▼
          Streamlit Dashboard
```

---

# 📂 Project Structure

```text
AquaPredict/
│
├── app.py
├── requirements.txt
├── packages.txt
├── runtime.txt
├── README.md
├── .gitignore
│
├── data/
│   └── water_potability.csv
│
├── models/
│   ├── scaler_model/
│   ├── random_forest/
│   ├── decision_tree/
│   ├── logistic_regression/
│   ├── naive_bayes/
│   └── metrics.json
│
└── src/
    ├── spark_session.py
    ├── preprocessing.py
    ├── spark_sql.py
    ├── rdd_operations.py
    ├── train_models.py
    └── predict.py
```

---

# 📊 Dataset

AquaPredict uses the **Water Potability Dataset**, which contains water quality measurements and a target variable indicating whether a water sample is suitable for drinking.

### Water Quality Parameters

| Parameter | Description |
|---|---|
| pH | Acidity or alkalinity of water |
| Hardness | Mineral content of water |
| Solids | Total dissolved solids |
| Chloramines | Chloramine concentration |
| Sulfate | Sulfate concentration |
| Conductivity | Electrical conductivity |
| Organic Carbon | Organic carbon concentration |
| Trihalomethanes | Trihalomethane concentration |
| Turbidity | Cloudiness of water |

### Target Variable

- `0` – Non-Potable Water
- `1` – Potable Water

---

# ⚙️ Installation and Setup

## Prerequisites

- Python 3.11
- Java JDK 17 (for local development)
- Git
- pip

## 1. Clone the Repository

```bash
git clone https://github.com/arambh8960/AquaPredict.git

cd AquaPredict
```

## 2. Create a Virtual Environment

```bash
python3 -m venv venv
```

## 3. Activate Virtual Environment

**Mac/Linux**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## 5. Run the Application

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

# ⚡ Apache Spark Configuration

The project uses PySpark to initialize a local Spark session.

The Spark session is configured with:

- Local execution mode: `local[*]`
- Spark Driver Memory: `2g`
- Shuffle Partitions: `4`
- Console Progress: Disabled

Spark MLlib is used for model training and prediction.

---

# 🔄 How It Works

### 1️⃣ Dataset Loading

The water quality dataset is loaded for analysis and processing.

### 2️⃣ Data Preprocessing

Missing values are handled and the dataset is prepared for machine learning.

### 3️⃣ Feature Engineering

Relevant water quality parameters are assembled into feature vectors.

### 4️⃣ Feature Scaling

A trained MinMaxScaler model is used to scale the input features.

### 5️⃣ Model Prediction

The input features are passed to the selected Spark MLlib classification model.

### 6️⃣ Prediction Result

The model predicts whether the water sample is potable or non-potable.

### 7️⃣ Visualization and Analysis

The Streamlit dashboard presents model comparisons and dataset analytics.

---

# 📈 Machine Learning Workflow

```text
Water Quality Dataset
        │
        ▼
Data Preprocessing
        │
        ▼
Feature Engineering
        │
        ▼
Feature Scaling
        │
        ▼
Model Training
        │
        ▼
Model Evaluation
        │
        ▼
Saved ML Models
        │
        ▼
Water Potability Prediction
```

---

# ☁️ Deployment

AquaPredict is deployed using **Streamlit Community Cloud**.

### Deployment Configuration

| Configuration | Value |
|---|---|
| Platform | Streamlit Community Cloud |
| GitHub Repository | arambh8960/AquaPredict |
| Branch | main |
| Main File | app.py |
| Python Version | 3.11 |
| Application Framework | Streamlit |

The project includes:

- `requirements.txt` – Python dependencies.
- `packages.txt` – System-level dependencies.
- `runtime.txt` – Runtime configuration file.

🌐 **Try the Live Application:**

https://aquapredict8960.streamlit.app/

---

# 🌟 Key Highlights

✅ Apache Spark Integration

✅ PySpark Data Processing

✅ Spark SQL and RDD Operations

✅ Spark MLlib Classification

✅ Four Machine Learning Algorithms

✅ Water Potability Prediction

✅ Feature Scaling and Preprocessing

✅ Model Performance Comparison

✅ Interactive Streamlit Dashboard

✅ Data Analytics and Visualization

✅ Cloud Deployment

---

# 🚀 Future Scope

- Real-time water quality monitoring.
- Integration with IoT-based water sensors.
- Advanced machine learning algorithms.
- More detailed water quality analysis.
- Improved model optimization.
- Integration of additional water quality datasets.
- Advanced analytics and reporting.

---

# 🎯 Project Objective

AquaPredict aims to demonstrate how **Big Data Processing and Machine Learning** can be combined to analyze water quality data and predict water potability.

The project provides an interactive platform that brings together Apache Spark, MLlib, and Streamlit for practical water quality analysis.

---

# 👨‍💻 Developer

### Arambh Tiwari

B.Tech CSE | Python Developer | Machine Learning & Big Data Enthusiast

**GitHub:** https://github.com/arambh8960

---

<div align="center">

⭐ If you found this project interesting, please consider giving it a star! ⭐

**Built with Python, Apache Spark, Machine Learning and Streamlit.**

</div>
