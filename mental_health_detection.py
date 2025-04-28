import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import plotly.express as px

# Load the dataset
def load_dataset(file_path):
    """
    Load the Mental Health & Technology Usage Dataset
    """
    df = pd.read_csv(r"C:\Users\DELL\Downloads\Machine learning projects\Mental Health Detection\archive\mental_health_and_technology_usage_2024.csv")
    print(f"Dataset loaded with {df.shape[0]} records and {df.shape[1]} features")
    return df

# Data exploration and cleaning
def explore_and_clean(df):
    """
    Explore the dataset and perform initial cleaning
    """
    # Display basic information
    print("Dataset overview:")
    print(df.info())
    print("\nSummary statistics:")
    print(df.describe())
    
    # Check for missing values
    print("\nMissing values per column:")
    print(df.isnull().sum())
    
    # Handle missing values if any
    if df.isnull().sum().sum() > 0:
        # For numeric columns, fill with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
    
    # Check for outliers in screen time
    Q1 = df['Screen_Time_Hours'].quantile(0.25)
    Q3 = df['Screen_Time_Hours'].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df['Screen_Time_Hours'] < lower_bound) | (df['Screen_Time_Hours'] > upper_bound)]
    print(f"\nFound {len(outliers)} outliers in Screen_Time_Hours")
    
    # For this analysis, we'll keep outliers but flag them
    df['Screen_Time_Outlier'] = ((df['Screen_Time_Hours'] < lower_bound) | 
                               (df['Screen_Time_Hours'] > upper_bound)).astype(int)
    
    # Check categorical variables
    print("\nCategorical variables:")
    for col in df.select_dtypes(include=['object']).columns:
        print(f"\n{col} value counts:")
        print(df[col].value_counts())
    
    return df

# Feature engineering
def engineer_features(df):
    """
    Create additional features from the existing dataset
    """
    # Total technology time (sum of different tech usages)
    df['Total_Tech_Hours'] = df['Technology_Usage_Hours'] + df['Social_Media_Usage_Hours'] + df['Gaming_Hours']
    
    # Calculate ratios
    df['Social_Media_Ratio'] = df['Social_Media_Usage_Hours'] / df['Total_Tech_Hours']
    df['Gaming_Ratio'] = df['Gaming_Hours'] / df['Total_Tech_Hours']
    
    # Create numeric encodings for categorical variables
    # Mental Health Status
    mental_health_mapping = {
        'Excellent': 5,
        'Good': 4,
        'Fair': 3,
        'Poor': 2,
        'Very Poor': 1
    }
    
    # If the values are different, we'll need to adjust this mapping
    if 'Mental_Health_Status' in df.columns:
        unique_values = df['Mental_Health_Status'].unique()
        if set(unique_values) != set(mental_health_mapping.keys()):
            print("Note: Mental health status values don't match expected values.")
            print(f"Actual values: {unique_values}")
        
        # Create numeric mental health score
        df['Mental_Health_Score'] = df['Mental_Health_Status'].map(mental_health_mapping)
        
        # Fill missing mappings with median if any
        if df['Mental_Health_Score'].isnull().sum() > 0:
            median_score = df['Mental_Health_Score'].median()
            df['Mental_Health_Score'] = df['Mental_Health_Score'].fillna(median_score)
    
    # Stress Level mapping
    stress_mapping = {
        'Very High': 5,
        'High': 4,
        'Moderate': 3,
        'Low': 2,
        'Very Low': 1
    }
    
    # If the values are different, we'll need to adjust this mapping
    if 'Stress_Level' in df.columns:
        unique_values = df['Stress_Level'].unique()
        if set(unique_values) != set(stress_mapping.keys()):
            print("Note: Stress level values don't match expected values.")
            print(f"Actual values: {unique_values}")
        
        # Create numeric stress score
        df['Stress_Score'] = df['Stress_Level'].map(stress_mapping)
        
        # Fill missing mappings with median if any
        if df['Stress_Score'].isnull().sum() > 0:
            median_score = df['Stress_Score'].median()
            df['Stress_Score'] = df['Stress_Score'].fillna(median_score)
    
    # Support systems access
    support_mapping = {
        'Strong': 3,
        'Moderate': 2,
        'Limited': 1,
        'None': 0
    }
    
    if 'Support_Systems_Access' in df:
        df['Support_Score'] = df['Support_Systems_Access'].map(support_mapping)
        
        # Fill missing mappings with median if any
        if df['Support_Score'].isnull().sum() > 0:
            median_score = df['Support_Score'].median()
            df['Support_Score'] = df['Support_Score'].fillna(median_score)
    
    # Work environment impact
    work_mapping = {
        'Positive': 2,
        'Neutral': 1,
        'Negative': 0
    }
    
    if 'Work_Environment_Impact' in df:
        df['Work_Impact_Score'] = df['Work_Environment_Impact'].map(work_mapping)
        
        # Fill missing mappings with median if any
        if df['Work_Impact_Score'].isnull().sum() > 0:
            median_score = df['Work_Impact_Score'].median()
            df['Work_Impact_Score'] = df['Work_Impact_Score'].fillna(median_score)
    
    # Online support usage
    online_support_mapping = {
        'Frequently': 3,
        'Occasionally': 2,
        'Rarely': 1,
        'Never': 0
    }
    
    if 'Online_Support_Usage' in df:
        df['Online_Support_Score'] = df['Online_Support_Usage'].map(online_support_mapping)
        
        # Fill missing mappings with median if any
        if df['Online_Support_Score'].isnull().sum() > 0:
            median_score = df['Online_Support_Score'].median()
            df['Online_Support_Score'] = df['Online_Support_Score'].fillna(median_score)
    
    # Create a wellbeing composite score
    # Higher score is better wellbeing
    if all(col in df.columns for col in ['Mental_Health_Score', 'Stress_Score', 'Sleep_Hours']):
        # Normalize sleep hours to 0-5 scale
        sleep_score = df['Sleep_Hours'] / 2  # Assuming 10 hours is max healthy sleep
        sleep_score = sleep_score.clip(0, 5)  # Cap at 5
        
        # Reverse stress score (5 is very high stress, so 5-stress_score means lower is better)
        reversed_stress = 6 - df['Stress_Score']  # Now higher is better
        
        # Combine: mental health (1-5) + reversed stress (1-5) + sleep (0-5)
        df['Wellbeing_Score'] = df['Mental_Health_Score'] + reversed_stress + sleep_score
        
        # Normalize to 0-10 scale
        max_possible = 15  # 5 + 5 + 5
        df['Wellbeing_Score'] = (df['Wellbeing_Score'] / max_possible) * 10
    
    # Create age groups
    bins = [0, 18, 25, 35, 50, 65, 100]
    labels = ['Under 18', '18-25', '26-35', '36-50', '51-65', 'Over 65']
    df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    
    # Screen time categories
    screen_bins = [0, 2, 4, 8, 24]
    screen_labels = ['Low', 'Moderate', 'High', 'Very High']
    df['Screen_Time_Category'] = pd.cut(df['Screen_Time_Hours'], bins=screen_bins, labels=screen_labels, right=False)
    
    # Create a risk index based on screen time, mental health, stress, and sleep
    if all(col in df.columns for col in ['Screen_Time_Hours', 'Mental_Health_Score', 'Stress_Score', 'Sleep_Hours']):
        # Risk factors:
        # - High screen time (normalize to 0-1, higher is worse)
        screen_risk = df['Screen_Time_Hours'] / 12  # Assuming 12+ hours is highest risk
        screen_risk = screen_risk.clip(0, 1)
        
        # - Poor mental health (convert to 0-1 scale, lower score is worse)
        mental_risk = (6 - df['Mental_Health_Score']) / 5  # Now 0=excellent, 1=very poor
        
        # - High stress (already 1-5, normalize to 0-1)
        stress_risk = (df['Stress_Score'] - 1) / 4  # Now 0=very low, 1=very high
        
        # - Poor sleep (less than 7 hours increases risk)
        sleep_risk = ((7 - df['Sleep_Hours']) / 7).clip(0, 1)  # 0=7+ hours, 1=0 hours
        
        # - Low physical activity
        activity_risk = ((5 - df['Physical_Activity_Hours']) / 5).clip(0, 1)  # 0=5+ hours, 1=0 hours
        
        # Combine risk factors (weighted average)
        df['Risk_Index'] = (
            0.25 * screen_risk + 
            0.25 * mental_risk + 
            0.20 * stress_risk + 
            0.15 * sleep_risk +
            0.15 * activity_risk
        ) * 10  # Scale to 0-10
        
        # Create risk categories
        conditions = [
            (df['Risk_Index'] < 3),
            (df['Risk_Index'] >= 3) & (df['Risk_Index'] < 5),
            (df['Risk_Index'] >= 5) & (df['Risk_Index'] < 7),
            (df['Risk_Index'] >= 7)
        ]
        values = ['Low Risk', 'Moderate Risk', 'High Risk', 'Severe Risk']
        df['Risk_Category'] = np.select(conditions, values, default='Unknown')
    
    return df

# Analyze relationships
def analyze_relationships(df):
    """
    Analyze relationships between technology usage and mental health indicators
    """
    # Calculate correlations between numeric variables
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    
    print("\nCorrelation matrix:")
    print(correlation_matrix)
    
    # Technology usage vs mental health
    tech_mental_corr = correlation_matrix.loc['Screen_Time_Hours', 'Mental_Health_Score'] if 'Mental_Health_Score' in correlation_matrix.columns else "N/A"
    tech_stress_corr = correlation_matrix.loc['Screen_Time_Hours', 'Stress_Score'] if 'Stress_Score' in correlation_matrix.columns else "N/A"
    tech_sleep_corr = correlation_matrix.loc['Screen_Time_Hours', 'Sleep_Hours'] if 'Sleep_Hours' in correlation_matrix.columns else "N/A"
    
    print(f"\nCorrelation between Screen Time and Mental Health Score: {tech_mental_corr}")
    print(f"Correlation between Screen Time and Stress Score: {tech_stress_corr}")
    print(f"Correlation between Screen Time and Sleep Hours: {tech_sleep_corr}")
    
    # Group analysis by screen time category
    if 'Screen_Time_Category' in df.columns and 'Mental_Health_Score' in df.columns:
        group_analysis = df.groupby('Screen_Time_Category').agg({
            'Mental_Health_Score': 'mean',
            'Stress_Score': 'mean',
            'Sleep_Hours': 'mean',
            'Wellbeing_Score': 'mean',
            'User_ID': 'count'
        }).rename(columns={'User_ID': 'Count'})
        
        print("\nMental health metrics by screen time category:")
        print(group_analysis)
    
    # Age group analysis
    if 'Age_Group' in df.columns:
        age_analysis = df.groupby('Age_Group').agg({
            'Screen_Time_Hours': 'mean',
            'Social_Media_Usage_Hours': 'mean',
            'Gaming_Hours': 'mean',
            'Mental_Health_Score': 'mean',
            'Stress_Score': 'mean',
            'Sleep_Hours': 'mean',
            'User_ID': 'count'
        }).rename(columns={'User_ID': 'Count'})
        
        print("\nMetrics by age group:")
        print(age_analysis)
    
    # Gender analysis
    if 'Gender' in df.columns:
        gender_analysis = df.groupby('Gender').agg({
            'Screen_Time_Hours': 'mean',
            'Social_Media_Usage_Hours': 'mean',
            'Gaming_Hours': 'mean',
            'Mental_Health_Score': 'mean',
            'Stress_Score': 'mean',
            'Sleep_Hours': 'mean',
            'User_ID': 'count'
        }).rename(columns={'User_ID': 'Count'})
        
        print("\nMetrics by gender:")
        print(gender_analysis)
    
    # Return analysis results for dashboard
    analysis_results = {
        'correlation_matrix': correlation_matrix
    }
    
    if 'Screen_Time_Category' in df.columns and 'Mental_Health_Score' in df.columns:
        analysis_results['screen_time_analysis'] = group_analysis
    
    if 'Age_Group' in df.columns:
        analysis_results['age_group_analysis'] = age_analysis
    
    if 'Gender' in df.columns:
        analysis_results['gender_analysis'] = gender_analysis
    
    return analysis_results

# Predictive modeling
def build_predictive_models(df):
    """
    Build predictive models for mental health based on usage patterns
    """
    model_results = {}
    
    # Model 1: Predicting Mental Health Score (if available)
    if 'Mental_Health_Score' in df.columns:
        print("\nBuilding model to predict Mental Health Score...")
        
        # Select features and target
        features = ['Age', 'Screen_Time_Hours', 'Social_Media_Usage_Hours', 
                   'Gaming_Hours', 'Sleep_Hours', 'Physical_Activity_Hours']
        
        # Only use features that exist in the dataframe
        X_features = [f for f in features if f in df.columns]
        X = df[X_features]
        y = df['Mental_Health_Score']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train a regression model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nFeature importance for Mental Health Score prediction:")
        print(feature_importance)
        
        # Calculate performance metrics
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        print(f"Model R² on training data: {train_score:.4f}")
        print(f"Model R² on test data: {test_score:.4f}")
        
        # Store results
        model_results['mental_health_model'] = {
            'model': model,
            'feature_importance': feature_importance,
            'features': X_features,
            'train_score': train_score,
            'test_score': test_score
        }
    
    # Model 2: Predicting Risk Category (if available)
    if 'Risk_Category' in df.columns:
        print("\nBuilding model to predict Risk Category...")
        
        # Select features and target
        features = ['Age', 'Screen_Time_Hours', 'Social_Media_Usage_Hours', 
                   'Gaming_Hours', 'Sleep_Hours', 'Physical_Activity_Hours',
                   'Mental_Health_Score', 'Stress_Score']
        
        # Only use features that exist in the dataframe
        X_features = [f for f in features if f in df.columns]
        X = df[X_features]
        y = df['Risk_Category']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train a classification model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nFeature importance for Risk Category prediction:")
        print(feature_importance)
        
        # Calculate performance metrics
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        print(f"Model accuracy on training data: {train_score:.4f}")
        print(f"Model accuracy on test data: {test_score:.4f}")
        
        # Store results
        model_results['risk_category_model'] = {
            'model': model,
            'feature_importance': feature_importance,
            'features': X_features,
            'train_score': train_score,
            'test_score': test_score
        }
    
    return model_results

# Prepare data for dashboard
def prepare_dashboard_data(df, analysis_results, model_results):
    """
    Prepare processed data for the dashboard
    """
    # Create a dictionary with all the necessary data
    dashboard_data = {
        'processed_data': df,
        'correlations': analysis_results.get('correlation_matrix'),
        'screen_time_impact': analysis_results.get('screen_time_analysis', None),
        'age_demographics': analysis_results.get('age_group_analysis', None),
        'gender_demographics': analysis_results.get('gender_analysis', None),
    }
    
    # Add model results if available
    if 'mental_health_model' in model_results:
        dashboard_data['mental_health_model'] = model_results['mental_health_model']
    
    if 'risk_category_model' in model_results:
        dashboard_data['risk_category_model'] = model_results['risk_category_model']
    
    return dashboard_data

# Main preprocessing function
def preprocess_mental_health_dataset(file_path, output_path=None):
    """
    Main function to preprocess the Mental Health & Technology Usage Dataset
    """
    # Load the dataset
    df = load_dataset(file_path)
    
    # Clean the data
    df = explore_and_clean(df)
    
    # Engineer features
    df = engineer_features(df)
    
    # Analyze relationships
    analysis_results = analyze_relationships(df)
    
    # Build predictive models
    model_results = build_predictive_models(df)
    
    # Prepare data for dashboard
    dashboard_data = prepare_dashboard_data(df, analysis_results, model_results)
    
    # Save processed data if path is provided
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")
    
    return dashboard_data

# Example usage
if __name__ == "__main__":
    # This would be the path to your dataset file
    file_path = "mental_health_tech_usage.csv"
    output_path = "processed_mental_health_data.csv"
    
    dashboard_data = preprocess_mental_health_dataset(file_path, output_path)
    print("Preprocessing complete. Data ready for dashboard visualization.")