# AquaPredict
## Water Quality Prediction Using Apache Spark and Machine Learning

AquaPredict is a water quality prediction system that uses Apache Spark and Machine Learning to classify water samples as Potable or Non-Potable.

### Technologies Used
- Python 3.11
- Apache Spark / PySpark 3.5.1
- Spark MLlib
- Streamlit
- Pandas, NumPy
- Plotly, Matplotlib, Seaborn

### Machine Learning Algorithms
1. Logistic Regression
2. Decision Tree
3. Naive Bayes
4. Random Forest

### Project Structure

AquaPredict/
- app.py
- requirements.txt
- README.md
- .gitignore
- data/
  - water_potability.csv
- models/
  - scaler_model/
  - random_forest/
  - decision_tree/
  - logistic_regression/
  - naive_bayes/
  - metrics.json
- src/
  - spark_session.py
  - preprocessing.py
  - spark_sql.py
  - rdd_operations.py
  - train_models.py
  - predict.py

### Prerequisites
- Python 3.11
- Java JDK 17
- Git

### Installation and Setup

1. Clone the repository:

   git clone https://github.com/arambh8960/AquaPredict.git

2. Open the project directory:

   cd AquaPredict

3. Create a virtual environment:

   python -m venv venv

4. Activate the virtual environment.

   Mac/Linux:

   source venv/bin/activate

   Windows:

   venv\Scripts\activate

5. Install dependencies:

   pip install -r requirements.txt

### Run the Application

   streamlit run app.py

Open the local URL displayed in your terminal.

### Dataset
Water Potability dataset containing nine water quality parameters and a Potability target column.

### Features
- Water potability prediction
- Four machine learning algorithms
- Model performance comparison
- Data visualization and analytics
- Apache Spark local processing

### Project Mode
Apache Spark runs in local mode using the available CPU cores.

