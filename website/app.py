# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load your trained models
try:
    risk_category_model = pickle.load(open('risk_category_model.pkl', 'rb'))
    mental_health_score_model = pickle.load(open('mental_health_score_model.pkl', 'rb'))
    model_loaded = True
    print("Models loaded successfully!")
except FileNotFoundError:
    print("Error: Model files not found. Ensure 'risk_category_model.pkl' and 'mental_health_score_model.pkl' are in the same directory as app.py.")
    risk_category_model = None
    mental_health_score_model = None
    model_loaded = False

# Define the expected input columns for both models
FEATURES_SCORE = ['screenTimeHours', 'socialMediaHours', 'gamingHours', 'sleepHours', 'physicalActivityHours', 'age']
FEATURES_CATEGORY = ['screenTimeHours', 'mentalHealthScore', 'physicalActivityHours', 'stressLevel', 'sleepHours', 'socialMediaHours', 'gamingHours', 'age']

# Mapping for categorical features (consistent with your preprocessing)
GENDER_MAP = {'Female': 0, 'Male': 1, 'Other': 2}
MENTAL_HEALTH_MAP = {'Excellent': 0, 'Good': 1, 'Fair': 2, 'Poor': 3}
STRESS_LEVEL_MAP = {'Low': 0, 'Medium': 1, 'High': 2}
SUPPORT_MAP = {'Yes': 1, 'No': 0}
WORK_IMPACT_MAP = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
ONLINE_SUPPORT_MAP = {'Yes': 1, 'No': 0}

def preprocess_input(data, features):
    df = pd.DataFrame([data])
    # Apply mappings
    df['gender'] = df['gender'].map(GENDER_MAP).fillna(-1)
    df['mentalHealthStatus'] = df['mentalHealthStatus'].map(MENTAL_HEALTH_MAP).fillna(-1)
    df['stressLevel'] = df['stressLevel'].map(STRESS_LEVEL_MAP).fillna(-1)
    df['supportSystemsAccess'] = df['supportSystemsAccess'].map(SUPPORT_MAP).fillna(-1)
    df['workEnvironmentImpact'] = df['workEnvironmentImpact'].map(WORK_IMPACT_MAP).fillna(-1)
    df['onlineSupportUsage'] = df['onlineSupportUsage'].map(ONLINE_SUPPORT_MAP).fillna(-1)
    # Ensure correct column order and select relevant features
    processed_data = df[features].values
    return processed_data

@app.route('/api/predict', methods=['POST'])
def predict():
    if not model_loaded:
        return jsonify({'error': 'Machine learning models not loaded'}), 500

    try:
        user_data = request.get_json()

        # Predict Mental Health Score
        processed_score_data = preprocess_input(user_data, FEATURES_SCORE)
        mental_health_score = mental_health_score_model.predict(processed_score_data)[0]

        # Update user_data with the predicted Mental Health Score for the next prediction
        user_data['mentalHealthScore'] = mental_health_score
        # Map stressLevel from string to numerical for prediction
        user_data['stressLevel'] = STRESS_LEVEL_MAP.get(user_data.get('stressLevel', 'Medium'), 1)

        # Predict Risk Category
        processed_category_data = preprocess_input(user_data, FEATURES_CATEGORY)
        risk_category = risk_category_model.predict(processed_category_data)[0]

        # Mock key factors and recommendations (you'll likely want to generate these based on the predictions)
        key_factors = [
            {'factor': 'Screen Time', 'impact': 'High', 'score': round(user_data.get('screenTimeHours', 0) * 1.2, 1)},
            {'factor': 'Physical Activity', 'impact': 'Medium', 'score': round(user_data.get('physicalActivityHours', 0) * 0.8, 1)},
            {'factor': 'Sleep Quality', 'impact': 'Medium', 'score': round(user_data.get('sleepHours', 0) * 0.9, 1)},
            {'factor': 'Social Media Usage', 'impact': 'Medium', 'score': round(user_data.get('socialMediaHours', 0) * 1.1, 1)},
        ]
        recommendations = generate_recommendations(user_data, {'riskCategory': risk_category})

        results = {
            'mentalHealthScore': float(mental_health_score),
            'riskCategory': risk_category,
            'keyFactors': key_factors,
            'recommendations': recommendations,
            'wellbeingScore': float(mental_health_score * 2) # Mock wellbeing score
        }
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def generate_recommendations(data, risk_results):
    """Generate personalized recommendations based on user data and risk assessment"""
    recommendations = []

    # Screen time recommendations
    if data.get('screenTimeHours', 0) > 6:
        recommendations.append("Consider reducing screen time to under 6 hours per day")

    # Physical activity recommendations
    if data.get('physicalActivityHours', 0) < 3:
        recommendations.append("Try to increase physical activity to at least 3 hours per week")

    # Sleep recommendations
    if data.get('sleepHours', 0) < 7:
        recommendations.append("Aim for at least 7 hours of sleep per night")

    # Social media recommendations
    if data.get('socialMediaHours', 0) > 3:
        recommendations.append("Consider limiting your time on social media")

    # Stress level recommendations
    stress_level = data.get('stressLevel')
    if stress_level == 'High':
        recommendations.append("Explore stress management techniques like meditation or yoga")
    elif stress_level == 'Medium':
        recommendations.append("Try to identify and manage sources of stress in your life")

    # Risk category specific recommendations
    if risk_results['riskCategory'] == 'Moderate':
        recommendations.append("Monitor your habits and consider small lifestyle adjustments")
    elif risk_results['riskCategory'] == 'High':
        recommendations.append("Consider seeking professional support for a mental health assessment")

    return recommendations

if __name__ == '__main__':
    app.run(debug=True)