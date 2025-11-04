"""
Advanced Model Evaluation and Comparison for Student Performance Prediction

This module provides comprehensive model evaluation, hyperparameter tuning,
and advanced ML techniques for student performance analysis.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import (train_test_split, cross_val_score, GridSearchCV, 
                                   RandomizedSearchCV, StratifiedKFold)
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor, 
                            GradientBoostingClassifier, GradientBoostingRegressor,
                            AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                           mean_squared_error, r2_score, mean_absolute_error,
                           roc_auc_score, precision_recall_curve, roc_curve)
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AdvancedModelEvaluator:
    """
    Comprehensive model evaluation and comparison system for student performance prediction.
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_models = {}
        self.feature_selector = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_and_preprocess_data(self, data_path):
        """Load and preprocess data for model training."""
        try:
            self.data = pd.read_csv(data_path)
            print(f"Data loaded: {self.data.shape}")
            
            # Handle categorical variables
            categorical_cols = ['gender', 'family_income', 'parent_education', 'learning_style']
            
            for col in categorical_cols:
                if col in self.data.columns:
                    self.label_encoders[col] = LabelEncoder()
                    self.data[col] = self.label_encoders[col].fit_transform(self.data[col])
            
            # Prepare features and targets
            feature_cols = [col for col in self.data.columns 
                          if col not in ['student_id', 'final_gpa', 'performance_category']]
            
            self.X = self.data[feature_cols]
            self.y_regression = self.data['final_gpa']
            self.y_classification = self.data['performance_category']
            
            print("Data preprocessing complete.")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
        
        return True
    
    def feature_selection_analysis(self):
        """Perform feature selection analysis."""
        print("Performing Feature Selection Analysis...")
        
        # Univariate feature selection
        selector = SelectKBest(score_func=f_classif, k=10)
        X_selected = selector.fit_transform(self.X, self.y_classification)
        
        # Get selected feature names
        selected_features = self.X.columns[selector.get_support()].tolist()
        feature_scores = selector.scores_
        
        # Create feature importance DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': self.X.columns,
            'score': feature_scores,
            'selected': selector.get_support()
        }).sort_values('score', ascending=False)
        
        print("\nTop 10 Features by Univariate Selection:")
        print(feature_importance_df.head(10))
        
        # Visualize feature importance
        plt.figure(figsize=(12, 8))
        top_features = feature_importance_df.head(10)
        plt.barh(range(len(top_features)), top_features['score'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance Score')
        plt.title('Top 10 Features by Univariate Selection')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('/workspace/feature_selection.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        self.feature_selector = selector
        return selected_features
    
    def define_model_grids(self):
        """Define models and their hyperparameter grids for tuning."""
        
        # Classification models
        classification_models = {
            'RandomForest_Clf': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'GradientBoosting_Clf': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 150],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'SVM_Clf': {
                'model': SVC(random_state=42, probability=True),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['rbf', 'poly'],
                    'gamma': ['scale', 'auto', 0.1, 1]
                }
            },
            'LogisticRegression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            },
            'KNN_Clf': {
                'model': KNeighborsClassifier(),
                'params': {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                }
            }
        }
        
        # Regression models
        regression_models = {
            'RandomForest_Reg': {
                'model': RandomForestRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'GradientBoosting_Reg': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 150],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'SVR': {
                'model': SVR(),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['rbf', 'poly'],
                    'gamma': ['scale', 'auto', 0.1, 1]
                }
            },
            'Ridge': {
                'model': Ridge(random_state=42),
                'params': {
                    'alpha': [0.01, 0.1, 1, 10, 100],
                    'solver': ['auto', 'svd', 'cholesky']
                }
            },
            'Lasso': {
                'model': Lasso(random_state=42),
                'params': {
                    'alpha': [0.01, 0.1, 1, 10, 100],
                    'max_iter': [1000, 2000]
                }
            }
        }
        
        return classification_models, regression_models
    
    def hyperparameter_tuning(self, task_type='both'):
        """Perform hyperparameter tuning for all models."""
        print("Starting Hyperparameter Tuning...")
        
        classification_models, regression_models = self.define_model_grids()
        
        # Split data
        X_train, X_test, y_reg_train, y_reg_test = train_test_split(
            self.X, self.y_regression, test_size=0.2, random_state=42
        )
        _, _, y_clf_train, y_clf_test = train_test_split(
            self.X, self.y_classification, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {'classification': {}, 'regression': {}}
        
        # Classification models
        if task_type in ['both', 'classification']:
            print("\nTuning Classification Models...")
            cv_clf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            for name, model_config in classification_models.items():
                print(f"Tuning {name}...")
                
                grid_search = RandomizedSearchCV(
                    model_config['model'],
                    model_config['params'],
                    n_iter=50,
                    cv=cv_clf,
                    scoring='accuracy',
                    random_state=42,
                    n_jobs=-1
                )
                
                grid_search.fit(X_train_scaled, y_clf_train)
                
                # Evaluate on test set
                best_model = grid_search.best_estimator_
                y_pred = best_model.predict(X_test_scaled)
                accuracy = accuracy_score(y_clf_test, y_pred)
                
                results['classification'][name] = {
                    'best_params': grid_search.best_params_,
                    'best_score': grid_search.best_score_,
                    'test_accuracy': accuracy,
                    'model': best_model
                }
                
                print(f"{name} - Best CV Score: {grid_search.best_score_:.4f}, Test Accuracy: {accuracy:.4f}")
        
        # Regression models
        if task_type in ['both', 'regression']:
            print("\nTuning Regression Models...")
            
            for name, model_config in regression_models.items():
                print(f"Tuning {name}...")
                
                grid_search = RandomizedSearchCV(
                    model_config['model'],
                    model_config['params'],
                    n_iter=50,
                    cv=5,
                    scoring='r2',
                    random_state=42,
                    n_jobs=-1
                )
                
                grid_search.fit(X_train_scaled, y_reg_train)
                
                # Evaluate on test set
                best_model = grid_search.best_estimator_
                y_pred = best_model.predict(X_test_scaled)
                r2 = r2_score(y_reg_test, y_pred)
                mse = mean_squared_error(y_reg_test, y_pred)
                
                results['regression'][name] = {
                    'best_params': grid_search.best_params_,
                    'best_score': grid_search.best_score_,
                    'test_r2': r2,
                    'test_mse': mse,
                    'model': best_model
                }
                
                print(f"{name} - Best CV Score: {grid_search.best_score_:.4f}, Test R²: {r2:.4f}")
        
        self.results = results
        return results
    
    def cross_validation_analysis(self):
        """Perform comprehensive cross-validation analysis."""
        print("Performing Cross-Validation Analysis...")
        
        if not self.results:
            print("No tuned models found. Run hyperparameter tuning first.")
            return
        
        cv_results = {'classification': {}, 'regression': {}}
        
        # Classification CV
        if 'classification' in self.results:
            cv_clf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
            X_scaled = self.scaler.fit_transform(self.X)
            
            for name, model_info in self.results['classification'].items():
                scores = cross_val_score(model_info['model'], X_scaled, 
                                       self.y_classification, cv=cv_clf, scoring='accuracy')
                cv_results['classification'][name] = {
                    'mean_score': scores.mean(),
                    'std_score': scores.std(),
                    'scores': scores
                }
        
        # Regression CV
        if 'regression' in self.results:
            X_scaled = self.scaler.fit_transform(self.X)
            
            for name, model_info in self.results['regression'].items():
                scores = cross_val_score(model_info['model'], X_scaled, 
                                       self.y_regression, cv=10, scoring='r2')
                cv_results['regression'][name] = {
                    'mean_score': scores.mean(),
                    'std_score': scores.std(),
                    'scores': scores
                }
        
        # Visualize CV results
        self._plot_cv_results(cv_results)
        
        return cv_results
    
    def _plot_cv_results(self, cv_results):
        """Plot cross-validation results."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Classification CV results
        if 'classification' in cv_results:
            clf_names = list(cv_results['classification'].keys())
            clf_scores = [cv_results['classification'][name]['scores'] 
                         for name in clf_names]
            
            axes[0].boxplot(clf_scores, labels=clf_names)
            axes[0].set_title('Classification Models - Cross-Validation Scores')
            axes[0].set_ylabel('Accuracy')
            axes[0].tick_params(axis='x', rotation=45)
        
        # Regression CV results
        if 'regression' in cv_results:
            reg_names = list(cv_results['regression'].keys())
            reg_scores = [cv_results['regression'][name]['scores'] 
                         for name in reg_names]
            
            axes[1].boxplot(reg_scores, labels=reg_names)
            axes[1].set_title('Regression Models - Cross-Validation Scores')
            axes[1].set_ylabel('R² Score')
            axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('/workspace/cv_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def model_comparison_report(self):
        """Generate comprehensive model comparison report."""
        print("="*80)
        print("COMPREHENSIVE MODEL COMPARISON REPORT")
        print("="*80)
        
        if not self.results:
            print("No results available. Run hyperparameter tuning first.")
            return
        
        # Classification results
        if 'classification' in self.results:
            print("\nCLASSIFICATION MODELS COMPARISON:")
            print("-" * 50)
            
            clf_comparison = []
            for name, result in self.results['classification'].items():
                clf_comparison.append({
                    'Model': name,
                    'CV Score': result['best_score'],
                    'Test Accuracy': result['test_accuracy'],
                    'Best Params': str(result['best_params'])[:50] + "..."
                })
            
            clf_df = pd.DataFrame(clf_comparison)
            clf_df = clf_df.sort_values('Test Accuracy', ascending=False)
            print(clf_df.to_string(index=False))
            
            # Best classification model
            best_clf = clf_df.iloc[0]
            print(f"\nBEST CLASSIFICATION MODEL: {best_clf['Model']}")
            print(f"Test Accuracy: {best_clf['Test Accuracy']:.4f}")
        
        # Regression results
        if 'regression' in self.results:
            print("\n\nREGRESSION MODELS COMPARISON:")
            print("-" * 50)
            
            reg_comparison = []
            for name, result in self.results['regression'].items():
                reg_comparison.append({
                    'Model': name,
                    'CV Score': result['best_score'],
                    'Test R²': result['test_r2'],
                    'Test MSE': result['test_mse'],
                    'Best Params': str(result['best_params'])[:50] + "..."
                })
            
            reg_df = pd.DataFrame(reg_comparison)
            reg_df = reg_df.sort_values('Test R²', ascending=False)
            print(reg_df.to_string(index=False))
            
            # Best regression model
            best_reg = reg_df.iloc[0]
            print(f"\nBEST REGRESSION MODEL: {best_reg['Model']}")
            print(f"Test R²: {best_reg['Test R²']:.4f}")
            print(f"Test MSE: {best_reg['Test MSE']:.4f}")
    
    def save_best_models(self):
        """Save the best performing models."""
        if not self.results:
            print("No models to save. Run hyperparameter tuning first.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Find and save best models
        if 'classification' in self.results:
            # Find best classification model
            best_clf_name = max(self.results['classification'], 
                              key=lambda x: self.results['classification'][x]['test_accuracy'])
            best_clf_model = self.results['classification'][best_clf_name]['model']
            
            # Save model
            joblib.dump(best_clf_model, f'/workspace/best_classification_model_{timestamp}.pkl')
            print(f"Best classification model ({best_clf_name}) saved.")
        
        if 'regression' in self.results:
            # Find best regression model
            best_reg_name = max(self.results['regression'], 
                              key=lambda x: self.results['regression'][x]['test_r2'])
            best_reg_model = self.results['regression'][best_reg_name]['model']
            
            # Save model
            joblib.dump(best_reg_model, f'/workspace/best_regression_model_{timestamp}.pkl')
            print(f"Best regression model ({best_reg_name}) saved.")
        
        # Save preprocessing objects
        joblib.dump(self.scaler, f'/workspace/scaler_{timestamp}.pkl')
        joblib.dump(self.label_encoders, f'/workspace/label_encoders_{timestamp}.pkl')
        print("Preprocessing objects saved.")
    
    def learning_curves(self):
        """Generate learning curves for best models."""
        from sklearn.model_selection import learning_curve
        
        if not self.results:
            print("No models available for learning curves.")
            return
        
        X_scaled = self.scaler.fit_transform(self.X)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Classification learning curve
        if 'classification' in self.results:
            best_clf_name = max(self.results['classification'], 
                              key=lambda x: self.results['classification'][x]['test_accuracy'])
            best_clf = self.results['classification'][best_clf_name]['model']
            
            train_sizes, train_scores, val_scores = learning_curve(
                best_clf, X_scaled, self.y_classification, cv=5, 
                train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
            )
            
            axes[0].plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training Score')
            axes[0].plot(train_sizes, np.mean(val_scores, axis=1), 'o-', label='Validation Score')
            axes[0].fill_between(train_sizes, np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                               np.mean(train_scores, axis=1) + np.std(train_scores, axis=1), alpha=0.1)
            axes[0].fill_between(train_sizes, np.mean(val_scores, axis=1) - np.std(val_scores, axis=1),
                               np.mean(val_scores, axis=1) + np.std(val_scores, axis=1), alpha=0.1)
            axes[0].set_xlabel('Training Set Size')
            axes[0].set_ylabel('Accuracy Score')
            axes[0].set_title(f'Learning Curve - {best_clf_name}')
            axes[0].legend()
            axes[0].grid(True)
        
        # Regression learning curve
        if 'regression' in self.results:
            best_reg_name = max(self.results['regression'], 
                              key=lambda x: self.results['regression'][x]['test_r2'])
            best_reg = self.results['regression'][best_reg_name]['model']
            
            train_sizes, train_scores, val_scores = learning_curve(
                best_reg, X_scaled, self.y_regression, cv=5, 
                train_sizes=np.linspace(0.1, 1.0, 10), scoring='r2', random_state=42
            )
            
            axes[1].plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training Score')
            axes[1].plot(train_sizes, np.mean(val_scores, axis=1), 'o-', label='Validation Score')
            axes[1].fill_between(train_sizes, np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                               np.mean(train_scores, axis=1) + np.std(train_scores, axis=1), alpha=0.1)
            axes[1].fill_between(train_sizes, np.mean(val_scores, axis=1) - np.std(val_scores, axis=1),
                               np.mean(val_scores, axis=1) + np.std(val_scores, axis=1), alpha=0.1)
            axes[1].set_xlabel('Training Set Size')
            axes[1].set_ylabel('R² Score')
            axes[1].set_title(f'Learning Curve - {best_reg_name}')
            axes[1].legend()
            axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('/workspace/learning_curves.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Main function to run advanced model evaluation."""
    print("Starting Advanced Model Evaluation...")
    
    # Initialize evaluator
    evaluator = AdvancedModelEvaluator()
    
    # Load data
    if not evaluator.load_and_preprocess_data('/workspace/student_data.csv'):
        # Generate data if not found
        from student_performance_analyzer import StudentPerformanceAnalyzer
        analyzer = StudentPerformanceAnalyzer()
        sample_data = analyzer.generate_sample_data(1000)
        sample_data.to_csv('/workspace/student_data.csv', index=False)
        evaluator.load_and_preprocess_data('/workspace/student_data.csv')
    
    # Feature selection
    print("\n1. Feature Selection Analysis...")
    selected_features = evaluator.feature_selection_analysis()
    
    # Hyperparameter tuning
    print("\n2. Hyperparameter Tuning...")
    results = evaluator.hyperparameter_tuning()
    
    # Cross-validation analysis
    print("\n3. Cross-Validation Analysis...")
    cv_results = evaluator.cross_validation_analysis()
    
    # Model comparison report
    print("\n4. Model Comparison Report...")
    evaluator.model_comparison_report()
    
    # Learning curves
    print("\n5. Learning Curves...")
    evaluator.learning_curves()
    
    # Save best models
    print("\n6. Saving Best Models...")
    evaluator.save_best_models()
    
    print("\nAdvanced model evaluation complete!")


if __name__ == "__main__":
    main()