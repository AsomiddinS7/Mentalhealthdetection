import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import plotly.express as px
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MentalHealthAnalyzer:
    def __init__(self, data_path: Union[str, Path]):
        self.data_path = Path(data_path)
        self.df = None
        self._load_data()
    
    def _load_data(self) -> None:
        """Load the dataset with error handling"""
        try:
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Dataset loaded successfully with {self.df.shape[0]} records and {self.df.shape[1]} features")
        except FileNotFoundError:
            logger.error(f"Data file not found at {self.data_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise
    
    def explore_and_clean(self) -> pd.DataFrame:
        """Explore the dataset and perform initial cleaning"""
        if self.df is None:
            raise ValueError("Data not loaded. Call _load_data() first.")
        
        logger.info("Starting data exploration and cleaning")
        
        # Basic information
        logger.info("Dataset overview:")
        logger.info(self.df.info())
        
        # Handle missing values efficiently
        missing_values = self.df.isnull().sum()
        if missing_values.sum() > 0:
            logger.info("Handling missing values")
            # Use fillna with method chaining for better performance
            self.df = self.df.fillna({
                col: self.df[col].median() 
                for col in self.df.select_dtypes(include=[np.number]).columns
            })
        
        # Detect and handle outliers more efficiently
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Create outlier flags
            self.df[f'{col}_Outlier'] = ((self.df[col] < lower_bound) | 
                                       (self.df[col] > upper_bound)).astype(int)
        
        return self.df
    
    def engineer_features(self) -> pd.DataFrame:
        """Create additional features from the existing dataset"""
        if self.df is None:
            raise ValueError("Data not loaded. Call _load_data() first.")
        
        logger.info("Starting feature engineering")
        
        # Define mappings as class attributes for reusability
        self._define_mappings()
        
        # Calculate total technology time
        tech_columns = ['Technology_Usage_Hours', 'Social_Media_Usage_Hours', 'Gaming_Hours']
        self.df['Total_Tech_Hours'] = self.df[tech_columns].sum(axis=1)
        
        # Calculate ratios efficiently
        self.df['Social_Media_Ratio'] = self.df['Social_Media_Usage_Hours'] / self.df['Total_Tech_Hours']
        self.df['Gaming_Ratio'] = self.df['Gaming_Hours'] / self.df['Total_Tech_Hours']
        
        # Apply mappings using vectorized operations
        self._apply_mappings()
        
        # Create composite scores
        self._create_composite_scores()
        
        return self.df
    
    def _define_mappings(self) -> None:
        """Define all categorical mappings"""
        self.mappings = {
            'Mental_Health_Status': {
                'Excellent': 5, 'Good': 4, 'Fair': 3, 'Poor': 2, 'Very Poor': 1
            },
            'Stress_Level': {
                'Very High': 5, 'High': 4, 'Moderate': 3, 'Low': 2, 'Very Low': 1
            },
            'Support_Systems_Access': {
                'Strong': 3, 'Moderate': 2, 'Limited': 1, 'None': 0
            },
            'Work_Environment_Impact': {
                'Positive': 2, 'Neutral': 1, 'Negative': 0
            },
            'Online_Support_Usage': {
                'Frequently': 3, 'Occasionally': 2, 'Rarely': 1, 'Never': 0
            }
        }
    
    def _apply_mappings(self) -> None:
        """Apply all categorical mappings efficiently"""
        for col, mapping in self.mappings.items():
            if col in self.df.columns:
                score_col = f"{col.split('_')[0]}_Score"
                self.df[score_col] = self.df[col].map(mapping)
                # Fill missing values with median
                self.df[score_col] = self.df[score_col].fillna(self.df[score_col].median())
    
    def _create_composite_scores(self) -> None:
        """Create composite scores for wellbeing and risk assessment"""
        # Wellbeing Score
        if all(col in self.df.columns for col in ['Mental_Health_Score', 'Stress_Score', 'Sleep_Hours']):
            sleep_score = (self.df['Sleep_Hours'] / 2).clip(0, 5)
            reversed_stress = 6 - self.df['Stress_Score']
            self.df['Wellbeing_Score'] = (self.df['Mental_Health_Score'] + reversed_stress + sleep_score) / 15 * 10
        
        # Risk Index
        if all(col in self.df.columns for col in ['Screen_Time_Hours', 'Mental_Health_Score', 'Stress_Score', 'Sleep_Hours']):
            risk_factors = {
                'screen_risk': (self.df['Screen_Time_Hours'] / 12).clip(0, 1),
                'mental_risk': (6 - self.df['Mental_Health_Score']) / 5,
                'stress_risk': (self.df['Stress_Score'] - 1) / 4,
                'sleep_risk': ((7 - self.df['Sleep_Hours']) / 7).clip(0, 1)
            }
            
            weights = [0.25, 0.25, 0.20, 0.15, 0.15]
            self.df['Risk_Index'] = sum(w * risk for w, risk in zip(weights, risk_factors.values())) * 10
            
            # Risk categories
            conditions = [
                (self.df['Risk_Index'] < 3),
                (self.df['Risk_Index'] >= 3) & (self.df['Risk_Index'] < 5),
                (self.df['Risk_Index'] >= 5) & (self.df['Risk_Index'] < 7),
                (self.df['Risk_Index'] >= 7)
            ]
            values = ['Low Risk', 'Moderate Risk', 'High Risk', 'Severe Risk']
            self.df['Risk_Category'] = np.select(conditions, values, default='Unknown')
    
    def analyze_relationships(self) -> pd.DataFrame:
        """Analyze relationships between variables"""
        if self.df is None:
            raise ValueError("Data not loaded. Call _load_data() first.")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.df[numeric_cols].corr()
        
        logger.info("Correlation analysis completed")
        return correlation_matrix

    def build_predictive_models(self) -> Dict[str, Union[RandomForestRegressor, RandomForestClassifier]]:
        """Build predictive models for mental health analysis"""
        if self.df is None:
            raise ValueError("Data not loaded. Call _load_data() first.")
        
        logger.info("Building predictive models")
        
        # Prepare features and target variables
        feature_cols = [
            'Screen_Time_Hours', 'Social_Media_Usage_Hours', 'Gaming_Hours',
            'Sleep_Hours', 'Physical_Activity_Hours', 'Stress_Score',
            'Support_Score', 'Work_Impact_Score', 'Online_Support_Score'
        ]
        
        target_cols = ['Mental_Health_Score', 'Wellbeing_Score', 'Risk_Index']
        
        # Ensure all required columns exist
        missing_features = [col for col in feature_cols if col not in self.df.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Prepare data
        X = self.df[feature_cols]
        models = {}
        
        for target in target_cols:
            if target not in self.df.columns:
                logger.warning(f"Skipping {target} as it's not in the dataset")
                continue
            
            y = self.df[target]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Choose model based on target type
            if target in ['Mental_Health_Score', 'Wellbeing_Score']:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Store model and scaler
            models[target] = {
                'model': model,
                'scaler': scaler,
                'feature_importance': dict(zip(feature_cols, model.feature_importances_))
            }
            
            logger.info(f"Model for {target} trained successfully")
        
        return models
    
    def prepare_dashboard_data(self) -> Dict:
        """Prepare data for dashboard visualization"""
        if self.df is None:
            raise ValueError("Data not loaded. Call _load_data() first.")
        
        logger.info("Preparing dashboard data")
        
        # Basic statistics
        dashboard_data = {
            'basic_stats': {
                'total_users': len(self.df),
                'avg_screen_time': self.df['Screen_Time_Hours'].mean(),
                'avg_mental_health': self.df['Mental_Health_Score'].mean(),
                'avg_wellbeing': self.df['Wellbeing_Score'].mean(),
                'avg_risk': self.df['Risk_Index'].mean()
            }
        }
        
        # Risk distribution
        if 'Risk_Category' in self.df.columns:
            risk_dist = self.df['Risk_Category'].value_counts().to_dict()
            dashboard_data['risk_distribution'] = risk_dist
        
        # Age group analysis
        if 'Age_Group' in self.df.columns:
            age_analysis = self.df.groupby('Age_Group').agg({
                'Screen_Time_Hours': 'mean',
                'Mental_Health_Score': 'mean',
                'Wellbeing_Score': 'mean',
                'Risk_Index': 'mean'
            }).to_dict()
            dashboard_data['age_analysis'] = age_analysis
        
        # Correlation matrix
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.df[numeric_cols].corr()
        dashboard_data['correlation_matrix'] = correlation_matrix.to_dict()
        
        logger.info("Dashboard data preparation completed")
        return dashboard_data

def main():
    """Main function to run the analysis"""
    try:
        # Initialize analyzer
        analyzer = MentalHealthAnalyzer('processed_mental_health_data.csv')
        
        # Perform analysis
        analyzer.explore_and_clean()
        analyzer.engineer_features()
        analyzer.analyze_relationships()
        
        # Build models
        models = analyzer.build_predictive_models()
        
        # Prepare dashboard data
        dashboard_data = analyzer.prepare_dashboard_data()
        
        logger.info("Analysis completed successfully")
        return models, dashboard_data
        
    except Exception as e:
        logger.error(f"Error in main analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()