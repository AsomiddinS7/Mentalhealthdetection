# app.py
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, timedelta
import requests
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize Mistral AI
MISTRAL_API_KEY = "YOUR_MISTRAL_API_KEY"  # Replace with your actual API key
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

# Load dataset for reference ranges and recommendations
try:
    dataset = pd.read_csv('mental_health_dataset.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Creating reference dataset...")
    # Create a reference dataset based on research
    dataset = pd.DataFrame({
        'screen_time_threshold': [4, 6, 8],
        'physical_activity_threshold': [2, 4, 6],
        'sleep_threshold': [6, 7, 8],
        'social_media_threshold': [2, 3, 4],
        'stress_level_threshold': [2, 3, 4],
        'risk_level': ['Low', 'Moderate', 'High']
    })
    dataset.to_csv('mental_health_dataset.csv', index=False)

def get_mistral_analysis(user_data, risk_category, key_factors):
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    # Prepare the context with research-based insights
    context = {
        "model": "mistral-tiny",
        "messages": [
            {"role": "system", "content": """You are a mental health analysis expert. Analyze the data and provide evidence-based recommendations using the following research guidelines:
            1. Screen time: >6 hours/day indicates high risk (WHO guidelines)
            2. Physical activity: <2 hours/day indicates high risk (CDC guidelines)
            3. Sleep: <6 hours/day indicates high risk (Sleep Foundation research)
            4. Social media: >3 hours/day indicates high risk (Mental Health Foundation studies)
            5. Stress levels: Impact on mental and physical health (APA research)"""},
            {"role": "user", "content": f"""Based on the following assessment data:
            - Risk Category: {risk_category}
            - Key Factors: {json.dumps(key_factors, indent=2)}
            - Screen Time: {user_data['screenTimeHours']} hours
            - Social Media: {user_data['socialMediaHours']} hours
            - Sleep: {user_data['sleepHours']} hours
            - Physical Activity: {user_data['physicalActivityHours']} hours
            - Stress Level: {user_data['stressLevel']}
            
            Provide a comprehensive analysis including:
            1. Evidence-based mental health assessment
            2. Specific areas requiring attention
            3. Research-backed recommendations
            4. Practical coping strategies
            5. Clear action steps
            
            Format the response with clear sections and bullet points."""}
        ],
        "temperature": 0.7,
        "max_tokens": 1000
    }

    try:
        response = requests.post(MISTRAL_API_URL, headers=headers, json=context)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            print(f"Error from Mistral AI: {response.text}")
            return get_fallback_analysis(user_data, risk_category, key_factors)
    except Exception as e:
        print(f"Error calling Mistral AI: {str(e)}")
        return get_fallback_analysis(user_data, risk_category, key_factors)

def get_fallback_analysis(user_data, risk_category, key_factors):
    # Provide research-based analysis when AI is unavailable
    analysis = []
    
    # 1. Evidence-based Assessment
    analysis.append("1. Evidence-based Mental Health Assessment:")
    assessment = get_evidence_based_assessment(user_data, risk_category)
    analysis.extend(assessment)

    # 2. Areas Requiring Attention
    analysis.append("\n2. Key Areas Requiring Attention:")
    concerns = get_research_based_concerns(key_factors)
    analysis.extend(concerns)

    # 3. Research-backed Recommendations
    analysis.append("\n3. Research-backed Recommendations:")
    recommendations = get_evidence_based_recommendations(user_data)
    analysis.extend(recommendations)

    # 4. Coping Strategies
    analysis.append("\n4. Evidence-based Coping Strategies:")
    strategies = get_research_based_strategies(risk_category)
    analysis.extend(strategies)

    # 5. Action Plan
    analysis.append("\n5. Actionable Steps:")
    steps = get_action_steps(risk_category, key_factors)
    analysis.extend(steps)

    return "\n".join(analysis)

def get_evidence_based_assessment(user_data, risk_category):
    assessment = []
    
    # Overall assessment based on WHO guidelines
    if risk_category == 'Low':
        assessment.append("According to WHO guidelines, your mental health indicators suggest positive well-being with good lifestyle habits.")
    elif risk_category == 'Moderate':
        assessment.append("Based on clinical research, your mental health shows moderate stress levels that would benefit from lifestyle modifications.")
    elif risk_category == 'High':
        assessment.append("Clinical indicators suggest significant stress levels requiring attention and possible professional support.")
    else:
        assessment.append("Your assessment indicates patterns associated with high stress and potential mental health risks. Professional consultation is recommended.")

    # Evidence-based factor analysis
    if user_data['screenTimeHours'] > 6:
        assessment.append("Research shows excessive screen time (>6 hours) can impact mental well-being and sleep quality.")
    if user_data['sleepHours'] < 7:
        assessment.append("Sleep duration below 7 hours is associated with increased stress and reduced cognitive function.")
    if user_data['physicalActivityHours'] < 2:
        assessment.append("Physical activity levels below WHO recommendations may impact mood and stress resilience.")

    return assessment

def get_research_based_concerns(key_factors):
    concerns = []
    high_impact_factors = [f for f in key_factors if f['impact'] == 'High']
    
    for factor in high_impact_factors:
        if factor['factor'] == 'Screen Time':
            concerns.append("- Extended screen time may lead to digital eye strain and disrupted sleep patterns")
        elif factor['factor'] == 'Sleep Quality':
            concerns.append("- Insufficient sleep is linked to increased anxiety and reduced stress coping ability")
        elif factor['factor'] == 'Physical Activity':
            concerns.append("- Limited physical activity can impact mood regulation and stress management")
        elif factor['factor'] == 'Social Media Usage':
            concerns.append("- Excessive social media use is associated with increased anxiety and FOMO")
        elif factor['factor'] == 'Stress Management':
            concerns.append("- Elevated stress levels can affect both mental and physical health")

    return concerns if concerns else ["No critical areas of concern based on current research guidelines."]

def get_evidence_based_recommendations(user_data):
    recommendations = []
    
    # Screen time recommendations (based on ophthalmology research)
    if user_data['screenTimeHours'] > 6:
        recommendations.extend([
            "- Implement the scientifically-proven 20-20-20 rule",
            "- Use blue light filters during evening hours",
            "- Set up regular screen breaks using time-tracking apps"
        ])
    
    # Sleep recommendations (based on sleep research)
    if user_data['sleepHours'] < 7:
        recommendations.extend([
            "- Maintain consistent sleep-wake cycles",
            "- Create a sleep-conducive environment (18-22°C, dark, quiet)",
            "- Practice relaxation techniques before bedtime"
        ])
    
    # Physical activity recommendations (based on WHO guidelines)
    if user_data['physicalActivityHours'] < 2:
        recommendations.extend([
            "- Incorporate 150 minutes of moderate exercise weekly",
            "- Break up sitting time every 30-60 minutes",
            "- Include both cardio and strength training activities"
        ])

    return recommendations

def get_research_based_strategies(risk_category):
    strategies = [
        "- Practice evidence-based mindfulness techniques",
        "- Use structured problem-solving approaches",
        "- Maintain social connections (shown to buffer stress)",
        "- Engage in regular physical activity"
    ]
    
    if risk_category in ['High', 'Severe']:
        strategies.extend([
            "- Consider professional counseling or therapy",
            "- Learn and practice stress-reduction techniques",
            "- Develop a support network"
        ])
    
    return strategies

def get_action_steps(risk_category, key_factors):
    steps = []
    
    if risk_category in ['High', 'Severe']:
        steps.extend([
            "1. Schedule a mental health assessment with a professional",
            "2. Begin daily stress-tracking journal",
            "3. Implement recommended lifestyle changes gradually"
        ])
    else:
        steps.extend([
            "1. Monitor your mental well-being using this tool weekly",
            "2. Implement one new healthy habit each week",
            "3. Track progress and adjust strategies as needed"
        ])
    
    return steps

def calculate_key_factors(data):
    factors = []
    
    # Screen Time Impact (based on research showing >6 hours is high risk)
    screen_time_score = max(0, 10 - (data['screenTimeHours'] / 10 * 10))
    factors.append({
        'factor': 'Screen Time',
        'impact': 'High' if data['screenTimeHours'] > 6 else 'Medium' if data['screenTimeHours'] > 4 else 'Low',
        'score': round(screen_time_score, 1),
        'description': f"Your screen time of {data['screenTimeHours']} hours {'exceeds' if data['screenTimeHours'] > 6 else 'is within' if data['screenTimeHours'] > 4 else 'is below'} recommended limits"
    })

    # Physical Activity Impact (based on WHO guidelines)
    activity_score = (data['physicalActivityHours'] / 5 * 10)
    factors.append({
        'factor': 'Physical Activity',
        'impact': 'High' if data['physicalActivityHours'] < 2 else 'Medium' if data['physicalActivityHours'] < 4 else 'Low',
        'score': round(activity_score, 1),
        'description': f"Your physical activity level {'meets' if data['physicalActivityHours'] >= 4 else 'partially meets' if data['physicalActivityHours'] >= 2 else 'falls below'} WHO recommendations"
    })

    # Sleep Impact (based on sleep research)
    sleep_score = (data['sleepHours'] / 10 * 10)
    factors.append({
        'factor': 'Sleep Quality',
        'impact': 'High' if data['sleepHours'] < 6 else 'Medium' if data['sleepHours'] < 7 else 'Low',
        'score': round(sleep_score, 1),
        'description': f"Your sleep duration of {data['sleepHours']} hours {'is optimal' if data['sleepHours'] >= 7 else 'is slightly below' if data['sleepHours'] >= 6 else 'is significantly below'} recommended levels"
    })

    # Social Media Impact (based on mental health research)
    social_media_score = max(0, 10 - (data['socialMediaHours'] / 5 * 10))
    factors.append({
        'factor': 'Social Media Usage',
        'impact': 'High' if data['socialMediaHours'] > 3 else 'Medium' if data['socialMediaHours'] > 2 else 'Low',
        'score': round(social_media_score, 1),
        'description': f"Your social media usage of {data['socialMediaHours']} hours {'exceeds' if data['socialMediaHours'] > 3 else 'is within' if data['socialMediaHours'] > 2 else 'is below'} recommended limits"
    })

    # Stress Impact (based on psychological research)
    stress_score = max(0, 10 - (data['stressLevel'] * 2))
    factors.append({
        'factor': 'Stress Management',
        'impact': 'High' if data['stressLevel'] > 3 else 'Medium' if data['stressLevel'] > 2 else 'Low',
        'score': round(stress_score, 1),
        'description': f"Your stress level is {'high' if data['stressLevel'] > 3 else 'moderate' if data['stressLevel'] > 2 else 'low'}, indicating {'significant' if data['stressLevel'] > 3 else 'some' if data['stressLevel'] > 2 else 'minimal'} impact on mental health"
    })

    return factors

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        user_data = request.get_json()

        # Normalize input values with research-based limits
        normalized_data = {
            'screenTimeHours': min(max(user_data.get('screenTimeHours', 0), 0), 10),
            'socialMediaHours': min(max(user_data.get('socialMediaHours', 0), 0), 5),
            'gamingHours': min(max(user_data.get('gamingHours', 0), 0), 5),
            'sleepHours': min(max(user_data.get('sleepHours', 0), 0), 10),
            'physicalActivityHours': min(max(user_data.get('physicalActivityHours', 0), 0), 5),
            'mentalHealthStatus': min(max(user_data.get('mentalHealthStatus', 3), 1), 5),
            'stressLevel': min(max(user_data.get('stressLevel', 2), 1), 5),
            'supportSystemsAccess': min(max(user_data.get('supportSystemsAccess', 0.5), 0), 1),
            'workEnvironmentImpact': min(max(user_data.get('workEnvironmentImpact', 1), 0), 2),
            'onlineSupportUsage': min(max(user_data.get('onlineSupportUsage', 0.5), 0), 1),
            'age': min(max(user_data.get('age', 25), 18), 65)
        }

        # Calculate weighted mental health score using research-based weights
        weights = {
            'mentalHealthStatus': 0.25,  # Self-reported mental health (25%)
            'stressLevel': 0.20,         # Stress impact (20%)
            'sleepHours': 0.15,          # Sleep quality (15%)
            'physicalActivityHours': 0.15, # Physical activity (15%)
            'screenTimeHours': 0.10,     # Screen time (10%)
            'socialMediaHours': 0.10,    # Social media impact (10%)
            'supportSystemsAccess': 0.05  # Support system (5%)
        }

        # Calculate component scores (normalized to 0-100)
        scores = {
            'mentalHealthStatus': min(100, max(0, (normalized_data['mentalHealthStatus'] / 5) * 100)),
            'stressLevel': min(100, max(0, (5 - normalized_data['stressLevel']) / 4 * 100)),  # Invert stress score
            'sleepHours': min(100, max(0, (normalized_data['sleepHours'] / 8) * 100)),
            'physicalActivityHours': min(100, max(0, (normalized_data['physicalActivityHours'] / 2) * 100)),
            'screenTimeHours': min(100, max(0, ((10 - normalized_data['screenTimeHours']) / 10) * 100)),
            'socialMediaHours': min(100, max(0, ((5 - normalized_data['socialMediaHours']) / 5) * 100)),
            'supportSystemsAccess': min(100, max(0, normalized_data['supportSystemsAccess'] * 100))
        }

        # Calculate weighted score (ensure it's between 0-100)
        mental_health_score = min(100, max(0, sum(scores[key] * weights[key] for key in weights)))

        # Determine risk category based on score thresholds from research
        if mental_health_score >= 80:
            risk_category = 'Low'
        elif mental_health_score >= 60:
            risk_category = 'Moderate'
        elif mental_health_score >= 40:
            risk_category = 'High'
        else:
            risk_category = 'Severe'

        # Calculate key factors with proper normalization
        key_factors = [
            {
                'factor': 'Screen Time',
                'impact': 'High' if normalized_data['screenTimeHours'] > 6 else 'Medium' if normalized_data['screenTimeHours'] > 4 else 'Low',
                'score': min(10, max(0, 10 - (normalized_data['screenTimeHours'] / 10 * 10))),
                'description': f"Your screen time of {normalized_data['screenTimeHours']} hours {'exceeds' if normalized_data['screenTimeHours'] > 6 else 'is within' if normalized_data['screenTimeHours'] > 4 else 'is below'} recommended limits"
            },
            {
                'factor': 'Physical Activity',
                'impact': 'High' if normalized_data['physicalActivityHours'] < 2 else 'Medium' if normalized_data['physicalActivityHours'] < 4 else 'Low',
                'score': min(10, max(0, normalized_data['physicalActivityHours'] / 2 * 10)),
                'description': f"Your physical activity level {'meets' if normalized_data['physicalActivityHours'] >= 4 else 'partially meets' if normalized_data['physicalActivityHours'] >= 2 else 'falls below'} WHO recommendations"
            },
            {
                'factor': 'Sleep Quality',
                'impact': 'High' if normalized_data['sleepHours'] < 6 else 'Medium' if normalized_data['sleepHours'] < 7 else 'Low',
                'score': min(10, max(0, normalized_data['sleepHours'] / 8 * 10)),
                'description': f"Your sleep duration of {normalized_data['sleepHours']} hours {'is optimal' if normalized_data['sleepHours'] >= 7 else 'is slightly below' if normalized_data['sleepHours'] >= 6 else 'is significantly below'} recommended levels"
            },
            {
                'factor': 'Social Media Usage',
                'impact': 'High' if normalized_data['socialMediaHours'] > 3 else 'Medium' if normalized_data['socialMediaHours'] > 2 else 'Low',
                'score': min(10, max(0, 10 - (normalized_data['socialMediaHours'] / 5 * 10))),
                'description': f"Your social media usage of {normalized_data['socialMediaHours']} hours {'exceeds' if normalized_data['socialMediaHours'] > 3 else 'is within' if normalized_data['socialMediaHours'] > 2 else 'is below'} recommended limits"
            }
        ]

        # Get AI-enhanced analysis
        ai_analysis = get_mistral_analysis(normalized_data, risk_category, key_factors)

        results = {
            'mentalHealthScore': float(mental_health_score),
            'riskCategory': risk_category,
            'keyFactors': key_factors,
            'aiAnalysis': ai_analysis,
            'componentScores': scores
        }
        return jsonify(results)
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)