import pickle
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Load your processed data
df = pd.read_csv(r"C:\Users\DELL\Downloads\Machine learning projects\Mental Health Detection\processed_mental_health_data.csv")

# Define features and target for Mental Health Score prediction
features_score = ['Screen_Time_Hours', 'Social_Media_Usage_Hours', 'Gaming_Hours', 'Sleep_Hours', 'Physical_Activity_Hours', 'Age']
target_score = 'Mental_Health_Score'
X_score = df[features_score]
y_score = df[target_score]
model_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
model_regressor.fit(X_score, y_score)
pickle.dump(model_regressor, open('mental_health_score_model.pkl', 'wb'))

# Define features and target for Risk Category prediction (assuming you have a 'Risk_Category' column)
# You'll need to define your risk categories and train your classifier accordingly
# Example (assuming 'Risk_Index' is used to create categories):
bins = [0, 2, 4, 5]  # Example bins for Low, Medium, High risk
labels = ['Low', 'Medium', 'High']
df['Risk_Category'] = pd.cut(df['Risk_Index'], bins=bins, labels=labels, right=False)
# Add 'Unknown' as a category if it's not already there
if 'Unknown' not in df['Risk_Category'].cat.categories:
     df['Risk_Category'] = df['Risk_Category'].cat.add_categories(['Unknown'])
df['Risk_Category'] = df['Risk_Category'].fillna('Unknown') # Handle potential NaN values

features_category = ['Screen_Time_Hours', 'Mental_Health_Score', 'Physical_Activity_Hours', 'Stress_Score', 'Sleep_Hours', 'Social_Media_Usage_Hours', 'Gaming_Hours', 'Age']
target_category = 'Risk_Category'
X_category = df[features_category]
y_category = df[target_category]
model_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
model_classifier.fit(X_category, y_category)
pickle.dump(model_classifier, open('risk_category_model.pkl', 'wb'))