from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
try:
    artifacts = joblib.load('model.pkl')
    model = artifacts['model']
    scaler = artifacts['scaler']
    feature_names = artifacts['feature_names']
    global_importances = getattr(model, 'feature_importances_', None)
except FileNotFoundError:
    print("Error: model.pkl not found. Please run train_model.py first.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json
    
    # 1. Parse Inputs
    air_temp = float(data['air_temp'])
    process_temp = float(data['process_temp'])
    rot_speed = float(data['rot_speed'])
    torque = float(data['torque'])
    tool_wear = float(data['tool_wear'])

    # --- PHYSICS GUARDRAIL ---
    if rot_speed > 2500 and torque > 65:
        return jsonify({
            'probability': 100,
            'status': "❌ DATA ANOMALY DETECTED",
            'color': "#8a2be2", # Purple
            'action': "🚨 SYSTEM ERROR: Physically impossible telemetry detected (High Speed + High Torque).",
            'impacts': []
        })

    # 2. Engineering Features
    temp_diff = process_temp - air_temp
    power = rot_speed * torque
    wear_rate = tool_wear / (rot_speed + 1)

    input_df = pd.DataFrame([[
        air_temp, process_temp, rot_speed, torque, tool_wear, temp_diff, power, wear_rate
    ]], columns=feature_names)

    # 3. Model Inference
    scaled_input = scaler.transform(input_df)
    prob = model.predict_proba(scaled_input)[0][1]
    
    # 4. Status and Color Logic
    if prob >= 0.50:
        status = "⚠️ CRITICAL FAILURE RISK"
        color = "#ff4c4c"
    elif prob >= 0.20:
        status = "⚠️ WARNING: MONITOR SYSTEM"
        color = "#ffa500"
    else:
        status = "✅ SYSTEM NOMINAL"
        color = "#4CAF50"

    # 5. Prescriptive Guidance Logic
    if prob >= 0.50:
        if tool_wear > 180:
            action = "🔧 ACTION REQUIRED: Tool wear critical (>180min). Schedule immediate replacement to prevent Part Damage."
        elif temp_diff > 10:
            action = "❄️ ACTION REQUIRED: Heat dissipation failure detected (ΔT > 10K). Check coolant levels and ventilation."
        elif torque > 55:
            action = "⚙️ ACTION REQUIRED: Torque overload detected. Reduce spindle load immediately to prevent Motor Burnout."
        else:
            action = "🚨 CRITICAL: Multiple telemetry anomalies detected. Emergency stop recommended for full inspection."
    elif prob >= 0.20:
        action = "📋 RECOMMENDATION: System showing early signs of degradation. Perform preventative maintenance within 24 hours."
    else:
        action = "🟢 RECOMMENDATION: No maintenance required. Continue with scheduled production cycle."

    # 6. DYNAMIC Confidence Factors (Impact Calculation)
    top_impacts = []
    if global_importances is not None:
        local_deviations = np.abs(scaled_input[0])
        dynamic_impacts = global_importances * local_deviations
        
        total_impact = np.sum(dynamic_impacts)
        if total_impact > 0:
            dynamic_impacts = (dynamic_impacts / total_impact) * 100
        
        feats = sorted(zip(feature_names, dynamic_impacts), key=lambda x: x[1], reverse=True)
        
        for name, imp in feats[:3]:
            clean_name = name.split(' [')[0].replace('_', ' ').title()
            top_impacts.append({"name": clean_name, "value": int(imp)})

    return jsonify({
        'probability': round(prob * 100),
        'status': status,
        'color': color,
        'action': action,
        'impacts': top_impacts
    })

if __name__ == "__main__":
    app.run(debug=True)