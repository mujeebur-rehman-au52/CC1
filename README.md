# Student Performance Analysis ML System

A comprehensive machine learning system for analyzing student profiles and predicting learning performance using advanced ML techniques, data exploration, and visualization.

## 🎯 Project Overview

This project provides a complete ML pipeline for educational data analysis, featuring:
- Synthetic student data generation with realistic profiles
- Multiple ML models for both classification and regression tasks
- Advanced data exploration and statistical analysis
- Comprehensive model evaluation and hyperparameter tuning
- Interactive visualizations and dashboards
- Risk assessment for identifying at-risk students

## 🚀 Features

### 📊 Data Generation & Analysis
- **Synthetic Data Generation**: Creates realistic student profiles with 15+ features
- **Statistical Analysis**: Comprehensive exploratory data analysis
- **Correlation Analysis**: Identifies key performance indicators
- **Performance Segmentation**: Groups students by performance levels
- **Risk Assessment**: Identifies students at risk of poor performance

### 🤖 Machine Learning Models
- **Classification Models**: Random Forest, Gradient Boosting, SVM, Logistic Regression, KNN
- **Regression Models**: Random Forest, Gradient Boosting, SVR, Ridge, Lasso
- **Hyperparameter Tuning**: Automated optimization using RandomizedSearchCV
- **Cross-Validation**: 10-fold cross-validation for robust evaluation
- **Feature Selection**: Univariate feature selection and importance analysis

### 📈 Visualizations & Dashboards
- **Static Visualizations**: 15+ different charts and plots
- **Interactive Dashboards**: 3D plots and interactive correlation matrices
- **Learning Curves**: Model performance vs training data size
- **Feature Importance**: Identifies most influential factors
- **Performance Dashboards**: Comprehensive student analytics

### 🎯 Student Profile Features
- **Demographics**: Age, gender, family background
- **Academic History**: Previous GPA, attendance rate
- **Study Habits**: Study hours, learning style preferences
- **Lifestyle Factors**: Sleep hours, stress levels, technology comfort
- **Extracurricular**: Job status, sports participation
- **Socioeconomic**: Family income, parent education level

## 📋 Requirements

```bash
pandas==2.1.4
numpy==1.24.3
scikit-learn==1.3.2
matplotlib==3.8.2
seaborn==0.13.0
jupyter==1.0.0
plotly==5.17.0
xgboost==2.0.3
lightgbm==4.1.0
joblib==1.3.2
scipy==1.11.4
```

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd student-performance-analysis
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## 📖 Usage

### 1. Basic Student Performance Analysis
```python
from student_performance_analyzer import StudentPerformanceAnalyzer

# Initialize analyzer
analyzer = StudentPerformanceAnalyzer()

# Generate sample data
student_data = analyzer.generate_sample_data(n_students=1000)

# Preprocess data
X, y_regression, y_classification = analyzer.preprocess_data(student_data)

# Train models
analyzer.train_models(X, y_regression, y_classification)

# Generate insights
analyzer.generate_insights_report()
```

### 2. Advanced Data Exploration
```python
from data_exploration_notebook import StudentDataExplorer

# Initialize explorer
explorer = StudentDataExplorer('/workspace/student_data.csv')

# Run comprehensive analysis
explorer.basic_statistics()
explorer.correlation_analysis()
explorer.performance_segmentation()
explorer.cluster_analysis()
explorer.risk_assessment()
```

### 3. Model Evaluation & Comparison
```python
from model_evaluation import AdvancedModelEvaluator

# Initialize evaluator
evaluator = AdvancedModelEvaluator()
evaluator.load_and_preprocess_data('/workspace/student_data.csv')

# Feature selection
selected_features = evaluator.feature_selection_analysis()

# Hyperparameter tuning
results = evaluator.hyperparameter_tuning()

# Model comparison
evaluator.model_comparison_report()
```

### 4. Predict Student Performance
```python
# Example student profile
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

# Get predictions
predictions = analyzer.predict_student_performance(sample_student)
print(predictions)
```

## 📊 Generated Outputs

The system generates multiple output files:

### Data Files
- `student_data.csv` - Generated student dataset
- `best_classification_model_*.pkl` - Saved best classification model
- `best_regression_model_*.pkl` - Saved best regression model

### Visualizations
- `feature_importance_analysis.png` - Feature importance charts
- `performance_dashboard.png` - Comprehensive performance analysis
- `correlation_matrix.png` - Feature correlation heatmap
- `distribution_analysis.png` - Variable distribution plots
- `performance_segmentation.png` - Student performance segments
- `cluster_visualization.png` - Student clustering analysis
- `risk_assessment.png` - Risk factor analysis
- `cv_results.png` - Cross-validation results
- `learning_curves.png` - Model learning curves

### Interactive Dashboards
- `interactive_3d_plot.html` - 3D performance visualization
- `interactive_correlation.html` - Interactive correlation matrix
- `interactive_dashboard.html` - Comprehensive dashboard

## 🔍 Key Insights & Findings

The analysis typically reveals:

1. **Top Performance Predictors**:
   - Previous GPA (strongest predictor)
   - Attendance rate
   - Study hours per week
   - Stress levels (negative correlation)

2. **Risk Factors**:
   - Low attendance (<80%)
   - Insufficient study time (<5 hours/week)
   - High stress levels (>7/10)
   - Poor sleep habits (<6 hours)

3. **Demographic Patterns**:
   - Family income impact on performance
   - Parent education correlation
   - Technology comfort influence

## 🎯 Model Performance

Typical model performance metrics:
- **Best Classification Accuracy**: ~85-90%
- **Best Regression R²**: ~0.75-0.85
- **Cross-Validation Stability**: High across all models

## 🔧 Customization

### Adding New Features
1. Modify the `generate_sample_data()` method in `StudentPerformanceAnalyzer`
2. Update preprocessing in `preprocess_data()` method
3. Adjust feature lists in analysis modules

### Custom Models
1. Add new models to `define_model_grids()` in `AdvancedModelEvaluator`
2. Include hyperparameter grids for tuning
3. Update evaluation metrics as needed

### New Visualizations
1. Extend `create_performance_dashboard()` method
2. Add custom plots to exploration modules
3. Create new interactive visualizations

## 📈 Performance Recommendations

Based on analysis results:

1. **For Institutions**:
   - Implement attendance monitoring systems
   - Provide stress management programs
   - Offer study skills workshops
   - Develop technology training programs

2. **For Students**:
   - Maintain consistent attendance
   - Balance study hours effectively
   - Manage stress levels
   - Ensure adequate sleep

3. **For Interventions**:
   - Early identification of at-risk students
   - Targeted support programs
   - Personalized learning approaches
   - Regular progress monitoring

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- scikit-learn for machine learning capabilities
- pandas and numpy for data manipulation
- matplotlib, seaborn, and plotly for visualizations
- Educational research community for feature insights

## 📞 Support

For questions or issues, please open an issue in the repository or contact the development team.

---

**Note**: This system generates synthetic data for demonstration purposes. For production use with real student data, ensure compliance with educational privacy regulations and data protection laws.
