# 🏭 Predictive Maintenance OS (PHM)

An industrial-grade Prognostics & Health Management (PHM) dashboard. This project uses Machine Learning to analyze live machine telemetry and predict impending equipment failures *before* they happen, providing engineers with actionable prescriptive guidance.

## ✨ Key Features

* **🧠 Explainable AI (XAI):** Displays dynamic **Confidence Factors** (Local Feature Impact) so operators know exactly *why* the AI triggered an alert.
* **🛡️ Physics-Based Guardrails:** Built-in logic prevents impossible operational states (e.g., maximum speed combined with maximum torque) at both the UI and API levels to prevent the model from generating out-of-distribution (OOD) errors.
* **🔧 Prescriptive Diagnostics:** Doesn't just predict failure—it translates predictions into specific Maintenance Action Plans (e.g., "Schedule immediate tool replacement" vs. "Check coolant levels").
* **📊 Interactive Simulation:** Real-time, responsive UI with gradient sliders that visually indicate physical operational boundaries.
* **🌗 Modern UI/UX:** Responsive CSS Grid layout, Chart.js animated gauges, and a seamless Dark/Light mode toggle.

## 🛠️ Tech Stack

* **Backend:** Python, Flask
* **Machine Learning:** Scikit-Learn (Random Forest Classifier), Pandas, NumPy, Joblib
* **Frontend:** HTML5, CSS3, Vanilla JavaScript, Chart.js

## 📂 Project Structure

```text
PredictiveMaintenance/
│
├── app.py                 # Main Flask web server and real-time API logic
├── train_model.py         # ML pipeline: Feature engineering, model training, and evaluation
├── Dataset.py             # Utility script to load and inspect the telemetry data
├── ai4i2020.csv           # Local dataset (UCI Machine Learning Repository)
├── model.pkl              # Serialized Random Forest model, scaler, and features
├── .gitignore             # Git ignore configurations
│
└── templates/
    └── index.html         # Frontend interactive dashboard UI


🚀 Installation & Setup
Clone the repository:

Bash
git clone [https://github.com/YSuhaybArthus/predictive-maintenance-os.git](https://github.com/SuhaybArthusS/predictive-maintenance-os.git)
cd predictive-maintenance-os
Create a virtual environment (Recommended):

Bash
python -m venv .venv

# Windows:
.venv\Scripts\activate

# Mac/Linux:
source .venv/bin/activate
Install required dependencies:

Bash
pip install pandas numpy scikit-learn flask joblib
⚙️ Usage Guide
Step 1: Train the Model Before running the server, you must train the AI and generate the model.pkl file. This script handles data scaling and feature engineering automatically.

Bash
python train_model.py
Step 2: Start the Web Server Launch the Flask application to serve the model.

Bash
python app.py
Step 3: Access the Dashboard Open your web browser and navigate to:

http://127.0.0.1:5000/

Adjust the sliders to simulate different operating conditions and watch the AI react and update its Confidence Factors in real-time!

🧪 Dataset Origin
The data used to train this model originates from the AI4I 2020 Predictive Maintenance Dataset provided by the UCI Machine Learning Repository.

<img width="2392" height="1618" alt="image" src="https://github.com/user-attachments/assets/2bd61dec-2f3f-46b2-8a96-7899e6c31797" />

for queries : arthus1506@gmail.com
