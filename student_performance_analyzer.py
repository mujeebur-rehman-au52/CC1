"""
Student Performance Analyzer - ML Model for Student Profile and Learning Performance Analysis

This module provides a comprehensive ML solution to analyze student profiles and predict learning performance.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class StudentPerformanceAnalyzer:
    """
    A comprehensive ML analyzer for student performance prediction and analysis.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
        self.performance_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def generate_sample_data(self, n_students=1000):
        """
        Generate synthetic student data for demonstration purposes.
        
        Args:
            n_students (int): Number of students to generate
            
        Returns:
            pd.DataFrame: Generated student data
        """
        np.random.seed(42)
        
        # Student demographics
        ages = np.random.normal(20, 2, n_students).astype(int)
        ages = np.clip(ages, 18, 25)
        
        genders = np.random.choice(['Male', 'Female', 'Other'], n_students, p=[0.45, 0.5, 0.05])
        
        # Socioeconomic factors
        family_income = np.random.choice(['Low', 'Middle', 'High'], n_students, p=[0.3, 0.5, 0.2])
        parent_education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 
                                          n_students, p=[0.3, 0.4, 0.25, 0.05])
        
        # Academic background
        previous_gpa = np.random.normal(3.0, 0.5, n_students)
        previous_gpa = np.clip(previous_gpa, 2.0, 4.0)
        
        # Study habits and behavior
        study_hours_per_week = np.random.gamma(2, 5, n_students)
        study_hours_per_week = np.clip(study_hours_per_week, 0, 40)
        
        attendance_rate = np.random.beta(8, 2, n_students)
        
        # Learning preferences
        learning_style = np.random.choice(['Visual', 'Auditory', 'Kinesthetic', 'Reading'], n_students)
        
        # Technology usage
        tech_comfort = np.random.randint(1, 11, n_students)  # 1-10 scale
        
        # Extracurricular activities
        has_job = np.random.choice([0, 1], n_students, p=[0.6, 0.4])
        sports_participation = np.random.choice([0, 1], n_students, p=[0.7, 0.3])
        
        # Mental health and wellness indicators
        stress_level = np.random.randint(1, 11, n_students)  # 1-10 scale
        sleep_hours = np.random.normal(7, 1.5, n_students)
        sleep_hours = np.clip(sleep_hours, 4, 12)
        
        # Calculate performance metrics based on various factors
        performance_base = (
            previous_gpa * 0.3 +
            (study_hours_per_week / 10) * 0.25 +
            attendance_rate * 0.2 +
            (10 - stress_level) / 10 * 0.15 +
            (tech_comfort / 10) * 0.1
        )
        
        # Add some noise and normalize
        performance_base += np.random.normal(0, 0.2, n_students)
        performance_base = np.clip(performance_base, 0, 4)
        
        # Final GPA (target variable)
        final_gpa = performance_base
        
        # Performance categories
        performance_category = pd.cut(final_gpa, 
                                    bins=[0, 2.0, 3.0, 3.5, 4.0], 
                                    labels=['Poor', 'Fair', 'Good', 'Excellent'])
        
        # Create DataFrame
        data = pd.DataFrame({
            'student_id': range(1, n_students + 1),
            'age': ages,
            'gender': genders,
            'family_income': family_income,
            'parent_education': parent_education,
            'previous_gpa': np.round(previous_gpa, 2),
            'study_hours_per_week': np.round(study_hours_per_week, 1),
            'attendance_rate': np.round(attendance_rate, 3),
            'learning_style': learning_style,
            'tech_comfort': tech_comfort,
            'has_job': has_job,
            'sports_participation': sports_participation,
            'stress_level': stress_level,
            'sleep_hours': np.round(sleep_hours, 1),
            'final_gpa': np.round(final_gpa, 2),
            'performance_category': performance_category
        })
        
        return data
    
    def preprocess_data(self, data):
        """
        Preprocess the student data for ML models.
        
        Args:
            data (pd.DataFrame): Raw student data
            
        Returns:
            tuple: Processed features and targets
        """
        self.performance_data = data.copy()
        
        # Encode categorical variables
        categorical_cols = ['gender', 'family_income', 'parent_education', 'learning_style']
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                data[col] = self.label_encoders[col].fit_transform(data[col])
            else:
                data[col] = self.label_encoders[col].transform(data[col])
        
        # Prepare features (exclude target variables and ID)
        feature_cols = [col for col in data.columns if col not in ['student_id', 'final_gpa', 'performance_category']]
        X = data[feature_cols]
        
        # Targets for different types of analysis
        y_regression = data['final_gpa']  # For regression
        y_classification = data['performance_category']  # For classification
        
        return X, y_regression, y_classification
    
    def train_models(self, X, y_regression, y_classification):
        """
        Train multiple ML models for both regression and classification tasks.
        
        Args:
            X (pd.DataFrame): Features
            y_regression (pd.Series): Continuous target (GPA)
            y_classification (pd.Series): Categorical target (performance category)
        """
        # Split data
        X_train, X_test, y_reg_train, y_reg_test = train_test_split(
            X, y_regression, test_size=0.2, random_state=42
        )
        
        _, _, y_clf_train, y_clf_test = train_test_split(
            X, y_classification, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.X_train, self.X_test = X_train_scaled, X_test_scaled
        self.y_train, self.y_test = y_reg_train, y_reg_test
        
        # Regression models
        regression_models = {
            'RandomForest_Reg': RandomForestRegressor(n_estimators=100, random_state=42),
            'LinearRegression': LinearRegression(),
            'SVR': SVR(kernel='rbf'),
            'GradientBoosting_Reg': GradientBoostingClassifier()
        }
        
        # Classification models
        classification_models = {
            'RandomForest_Clf': RandomForestClassifier(n_estimators=100, random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM_Clf': SVC(random_state=42),
            'GradientBoosting_Clf': GradientBoostingClassifier(random_state=42)
        }
        
        # Train regression models
        print("Training Regression Models...")
        for name, model in regression_models.items():
            if name != 'GradientBoosting_Reg':  # Skip GB for regression
                model.fit(X_train_scaled, y_reg_train)
                y_pred = model.predict(X_test_scaled)
                mse = mean_squared_error(y_reg_test, y_pred)
                r2 = r2_score(y_reg_test, y_pred)
                
                self.models[name] = {
                    'model': model,
                    'type': 'regression',
                    'mse': mse,
                    'r2': r2,
                    'predictions': y_pred
                }
                print(f"{name} - MSE: {mse:.4f}, R²: {r2:.4f}")
        
        # Train classification models
        print("\nTraining Classification Models...")
        for name, model in classification_models.items():
            model.fit(X_train_scaled, y_clf_train)
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_clf_test, y_pred)
            
            self.models[name] = {
                'model': model,
                'type': 'classification',
                'accuracy': accuracy,
                'predictions': y_pred,
                'y_test': y_clf_test
            }
            print(f"{name} - Accuracy: {accuracy:.4f}")
    
    def analyze_feature_importance(self):
        """
        Analyze and visualize feature importance from tree-based models.
        """
        # Get feature names
        feature_names = ['age', 'gender', 'family_income', 'parent_education', 'previous_gpa',
                        'study_hours_per_week', 'attendance_rate', 'learning_style', 'tech_comfort',
                        'has_job', 'sports_participation', 'stress_level', 'sleep_hours']
        
        plt.figure(figsize=(15, 10))
        
        # Random Forest Regression Feature Importance
        plt.subplot(2, 2, 1)
        if 'RandomForest_Reg' in self.models:
            importance = self.models['RandomForest_Reg']['model'].feature_importances_
            indices = np.argsort(importance)[::-1]
            plt.bar(range(len(importance)), importance[indices])
            plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45)
            plt.title('Feature Importance - Random Forest Regression')
            plt.tight_layout()
        
        # Random Forest Classification Feature Importance
        plt.subplot(2, 2, 2)
        if 'RandomForest_Clf' in self.models:
            importance = self.models['RandomForest_Clf']['model'].feature_importances_
            indices = np.argsort(importance)[::-1]
            plt.bar(range(len(importance)), importance[indices])
            plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45)
            plt.title('Feature Importance - Random Forest Classification')
            plt.tight_layout()
        
        # Data distribution analysis
        plt.subplot(2, 2, 3)
        if self.performance_data is not None:
            self.performance_data['performance_category'].value_counts().plot(kind='bar')
            plt.title('Distribution of Performance Categories')
            plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 4)
        if self.performance_data is not None:
            plt.hist(self.performance_data['final_gpa'], bins=20, alpha=0.7)
            plt.title('Distribution of Final GPA')
            plt.xlabel('GPA')
            plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('/workspace/feature_importance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_performance_dashboard(self):
        """
        Create a comprehensive performance analysis dashboard.
        """
        if self.performance_data is None:
            print("No data available. Please run the analysis first.")
            return
        
        plt.figure(figsize=(20, 15))
        
        # 1. GPA vs Study Hours
        plt.subplot(3, 4, 1)
        plt.scatter(self.performance_data['study_hours_per_week'], self.performance_data['final_gpa'], alpha=0.6)
        plt.xlabel('Study Hours per Week')
        plt.ylabel('Final GPA')
        plt.title('GPA vs Study Hours')
        
        # 2. Attendance vs Performance
        plt.subplot(3, 4, 2)
        plt.scatter(self.performance_data['attendance_rate'], self.performance_data['final_gpa'], alpha=0.6)
        plt.xlabel('Attendance Rate')
        plt.ylabel('Final GPA')
        plt.title('GPA vs Attendance Rate')
        
        # 3. Stress Level Impact
        plt.subplot(3, 4, 3)
        stress_gpa = self.performance_data.groupby('stress_level')['final_gpa'].mean()
        plt.bar(stress_gpa.index, stress_gpa.values)
        plt.xlabel('Stress Level')
        plt.ylabel('Average GPA')
        plt.title('Stress Level vs Average GPA')
        
        # 4. Gender Distribution
        plt.subplot(3, 4, 4)
        gender_perf = self.performance_data.groupby('gender')['final_gpa'].mean()
        plt.bar(gender_perf.index, gender_perf.values)
        plt.xlabel('Gender')
        plt.ylabel('Average GPA')
        plt.title('Performance by Gender')
        plt.xticks(rotation=45)
        
        # 5. Family Income Impact
        plt.subplot(3, 4, 5)
        income_perf = self.performance_data.groupby('family_income')['final_gpa'].mean()
        plt.bar(income_perf.index, income_perf.values)
        plt.xlabel('Family Income')
        plt.ylabel('Average GPA')
        plt.title('Performance by Family Income')
        
        # 6. Parent Education Impact
        plt.subplot(3, 4, 6)
        parent_perf = self.performance_data.groupby('parent_education')['final_gpa'].mean()
        plt.bar(parent_perf.index, parent_perf.values)
        plt.xlabel('Parent Education')
        plt.ylabel('Average GPA')
        plt.title('Performance by Parent Education')
        plt.xticks(rotation=45)
        
        # 7. Learning Style Distribution
        plt.subplot(3, 4, 7)
        learning_perf = self.performance_data.groupby('learning_style')['final_gpa'].mean()
        plt.bar(learning_perf.index, learning_perf.values)
        plt.xlabel('Learning Style')
        plt.ylabel('Average GPA')
        plt.title('Performance by Learning Style')
        plt.xticks(rotation=45)
        
        # 8. Sleep Hours vs Performance
        plt.subplot(3, 4, 8)
        plt.scatter(self.performance_data['sleep_hours'], self.performance_data['final_gpa'], alpha=0.6)
        plt.xlabel('Sleep Hours')
        plt.ylabel('Final GPA')
        plt.title('GPA vs Sleep Hours')
        
        # 9. Technology Comfort Impact
        plt.subplot(3, 4, 9)
        tech_gpa = self.performance_data.groupby('tech_comfort')['final_gpa'].mean()
        plt.plot(tech_gpa.index, tech_gpa.values, marker='o')
        plt.xlabel('Technology Comfort Level')
        plt.ylabel('Average GPA')
        plt.title('Tech Comfort vs Performance')
        
        # 10. Job Impact
        plt.subplot(3, 4, 10)
        job_perf = self.performance_data.groupby('has_job')['final_gpa'].mean()
        plt.bar(['No Job', 'Has Job'], job_perf.values)
        plt.ylabel('Average GPA')
        plt.title('Employment Status vs Performance')
        
        # 11. Sports Participation
        plt.subplot(3, 4, 11)
        sports_perf = self.performance_data.groupby('sports_participation')['final_gpa'].mean()
        plt.bar(['No Sports', 'Sports'], sports_perf.values)
        plt.ylabel('Average GPA')
        plt.title('Sports Participation vs Performance')
        
        # 12. Age Distribution
        plt.subplot(3, 4, 12)
        plt.hist(self.performance_data['age'], bins=10, alpha=0.7)
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        plt.title('Age Distribution')
        
        plt.tight_layout()
        plt.savefig('/workspace/performance_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_student_performance(self, student_profile):
        """
        Predict performance for a new student profile.
        
        Args:
            student_profile (dict): Student characteristics
            
        Returns:
            dict: Predictions from different models
        """
        if not self.models:
            print("Models not trained yet. Please train models first.")
            return None
        
        # Convert profile to DataFrame
        profile_df = pd.DataFrame([student_profile])
        
        # Encode categorical variables
        categorical_cols = ['gender', 'family_income', 'parent_education', 'learning_style']
        for col in categorical_cols:
            if col in profile_df.columns and col in self.label_encoders:
                profile_df[col] = self.label_encoders[col].transform(profile_df[col])
        
        # Scale features
        profile_scaled = self.scaler.transform(profile_df)
        
        predictions = {}
        for name, model_info in self.models.items():
            if model_info['type'] == 'regression':
                pred = model_info['model'].predict(profile_scaled)[0]
                predictions[f"{name}_GPA"] = round(pred, 2)
            else:
                pred = model_info['model'].predict(profile_scaled)[0]
                predictions[f"{name}_Category"] = pred
        
        return predictions
    
    def generate_insights_report(self):
        """
        Generate a comprehensive insights report.
        """
        if self.performance_data is None:
            print("No data available for insights.")
            return
        
        print("=" * 60)
        print("STUDENT PERFORMANCE ANALYSIS INSIGHTS REPORT")
        print("=" * 60)
        
        # Basic statistics
        print("\n1. DATASET OVERVIEW:")
        print(f"Total Students: {len(self.performance_data)}")
        print(f"Average GPA: {self.performance_data['final_gpa'].mean():.2f}")
        print(f"GPA Standard Deviation: {self.performance_data['final_gpa'].std():.2f}")
        
        # Performance distribution
        print("\n2. PERFORMANCE DISTRIBUTION:")
        perf_dist = self.performance_data['performance_category'].value_counts()
        for category, count in perf_dist.items():
            percentage = (count / len(self.performance_data)) * 100
            print(f"{category}: {count} students ({percentage:.1f}%)")
        
        # Key correlations
        print("\n3. KEY CORRELATIONS WITH FINAL GPA:")
        numerical_cols = ['age', 'previous_gpa', 'study_hours_per_week', 'attendance_rate',
                         'tech_comfort', 'stress_level', 'sleep_hours']
        
        correlations = []
        for col in numerical_cols:
            corr = self.performance_data['final_gpa'].corr(self.performance_data[col])
            correlations.append((col, corr))
        
        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        for feature, corr in correlations:
            print(f"{feature}: {corr:.3f}")
        
        # Model performance summary
        print("\n4. MODEL PERFORMANCE SUMMARY:")
        for name, model_info in self.models.items():
            if model_info['type'] == 'regression':
                print(f"{name}: R² = {model_info['r2']:.3f}, MSE = {model_info['mse']:.3f}")
            else:
                print(f"{name}: Accuracy = {model_info['accuracy']:.3f}")
        
        # Insights and recommendations
        print("\n5. KEY INSIGHTS AND RECOMMENDATIONS:")
        
        # High-impact factors
        high_corr_features = [f for f, c in correlations if abs(c) > 0.3]
        print(f"\nHigh-impact factors (|correlation| > 0.3): {', '.join(high_corr_features)}")
        
        # Performance by demographics
        print(f"\nAverage GPA by Family Income:")
        income_gpa = self.performance_data.groupby('family_income')['final_gpa'].mean()
        for income, gpa in income_gpa.items():
            print(f"  {income}: {gpa:.2f}")
        
        print(f"\nAverage GPA by Parent Education:")
        parent_gpa = self.performance_data.groupby('parent_education')['final_gpa'].mean()
        for education, gpa in parent_gpa.items():
            print(f"  {education}: {gpa:.2f}")
        
        print("\n6. ACTIONABLE RECOMMENDATIONS:")
        print("• Focus on improving attendance rates (strong correlation with performance)")
        print("• Implement stress management programs (negative correlation with performance)")
        print("• Encourage optimal study hours (balance between too little and too much)")
        print("• Provide technology training to improve comfort levels")
        print("• Consider sleep education programs for better academic outcomes")
        print("• Develop targeted support for students from different socioeconomic backgrounds")


def main():
    """
    Main function to demonstrate the Student Performance Analyzer.
    """
    print("Initializing Student Performance Analyzer...")
    analyzer = StudentPerformanceAnalyzer()
    
    # Generate sample data
    print("Generating sample student data...")
    student_data = analyzer.generate_sample_data(n_students=1000)
    
    # Save the generated data
    student_data.to_csv('/workspace/student_data.csv', index=False)
    print("Sample data saved to 'student_data.csv'")
    
    # Preprocess data
    print("Preprocessing data...")
    X, y_regression, y_classification = analyzer.preprocess_data(student_data)
    
    # Train models
    print("Training ML models...")
    analyzer.train_models(X, y_regression, y_classification)
    
    # Analyze feature importance
    print("Analyzing feature importance...")
    analyzer.analyze_feature_importance()
    
    # Create performance dashboard
    print("Creating performance dashboard...")
    analyzer.create_performance_dashboard()
    
    # Generate insights report
    analyzer.generate_insights_report()
    
    # Example prediction
    print("\n" + "="*60)
    print("EXAMPLE STUDENT PERFORMANCE PREDICTION")
    print("="*60)
    
    sample_student = {
        'age': 20,
        'gender': 'Female',
        'family_income': 'Middle',
        'parent_education': 'Bachelor',
        'previous_gpa': 3.2,
        'study_hours_per_week': 15.0,
        'attendance_rate': 0.9,
        'learning_style': 'Visual',
        'tech_comfort': 8,
        'has_job': 0,
        'sports_participation': 1,
        'stress_level': 5,
        'sleep_hours': 7.5
    }
    
    predictions = analyzer.predict_student_performance(sample_student)
    if predictions:
        print("\nPredictions for sample student:")
        for model, prediction in predictions.items():
            print(f"{model}: {prediction}")
    
    print("\nAnalysis complete! Check the generated visualizations and data files.")


if __name__ == "__main__":
    main()