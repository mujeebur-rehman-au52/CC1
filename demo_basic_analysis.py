#!/usr/bin/env python3
"""
Basic Student Performance Analysis Demo

This is a simplified demonstration of the student performance analysis system
that works with minimal dependencies (only Python standard library).
"""

import csv
import random
import math
import json
from datetime import datetime

class SimpleStudentAnalyzer:
    """A simplified student performance analyzer using only standard library."""
    
    def __init__(self):
        self.students = []
        self.analysis_results = {}
        
    def generate_sample_data(self, n_students=100):
        """Generate sample student data using only standard library."""
        random.seed(42)
        
        students = []
        
        for i in range(1, n_students + 1):
            # Generate student profile
            student = {
                'student_id': i,
                'age': random.randint(18, 25),
                'gender': random.choice(['Male', 'Female', 'Other']),
                'family_income': random.choice(['Low', 'Middle', 'High']),
                'parent_education': random.choice(['High School', 'Bachelor', 'Master', 'PhD']),
                'previous_gpa': round(random.uniform(2.0, 4.0), 2),
                'study_hours_per_week': round(random.uniform(0, 40), 1),
                'attendance_rate': round(random.uniform(0.5, 1.0), 3),
                'learning_style': random.choice(['Visual', 'Auditory', 'Kinesthetic', 'Reading']),
                'tech_comfort': random.randint(1, 10),
                'has_job': random.choice([0, 1]),
                'sports_participation': random.choice([0, 1]),
                'stress_level': random.randint(1, 10),
                'sleep_hours': round(random.uniform(4, 12), 1)
            }
            
            # Calculate final GPA based on factors (simplified model)
            performance_base = (
                student['previous_gpa'] * 0.3 +
                (student['study_hours_per_week'] / 10) * 0.25 +
                student['attendance_rate'] * 0.2 +
                (10 - student['stress_level']) / 10 * 0.15 +
                (student['tech_comfort'] / 10) * 0.1
            )
            
            # Add some noise
            performance_base += random.uniform(-0.2, 0.2)
            student['final_gpa'] = round(max(0, min(4.0, performance_base)), 2)
            
            # Performance category
            if student['final_gpa'] < 2.0:
                student['performance_category'] = 'Poor'
            elif student['final_gpa'] < 3.0:
                student['performance_category'] = 'Fair'
            elif student['final_gpa'] < 3.5:
                student['performance_category'] = 'Good'
            else:
                student['performance_category'] = 'Excellent'
            
            students.append(student)
        
        self.students = students
        return students
    
    def save_data_csv(self, filename='demo_student_data.csv'):
        """Save student data to CSV file."""
        if not self.students:
            print("No data to save. Generate data first.")
            return
        
        fieldnames = self.students[0].keys()
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.students)
        
        print(f"Data saved to {filename}")
    
    def basic_statistics(self):
        """Calculate basic statistics."""
        if not self.students:
            print("No data available. Generate data first.")
            return
        
        print("="*60)
        print("BASIC STUDENT PERFORMANCE STATISTICS")
        print("="*60)
        
        # Basic counts
        total_students = len(self.students)
        print(f"\nTotal Students: {total_students}")
        
        # GPA statistics
        gpas = [s['final_gpa'] for s in self.students]
        avg_gpa = sum(gpas) / len(gpas)
        min_gpa = min(gpas)
        max_gpa = max(gpas)
        
        print(f"\nGPA Statistics:")
        print(f"  Average GPA: {avg_gpa:.2f}")
        print(f"  Minimum GPA: {min_gpa:.2f}")
        print(f"  Maximum GPA: {max_gpa:.2f}")
        
        # Performance distribution
        perf_counts = {}
        for student in self.students:
            category = student['performance_category']
            perf_counts[category] = perf_counts.get(category, 0) + 1
        
        print(f"\nPerformance Distribution:")
        for category, count in perf_counts.items():
            percentage = (count / total_students) * 100
            print(f"  {category}: {count} students ({percentage:.1f}%)")
        
        # Study hours analysis
        study_hours = [s['study_hours_per_week'] for s in self.students]
        avg_study_hours = sum(study_hours) / len(study_hours)
        print(f"\nAverage Study Hours per Week: {avg_study_hours:.1f}")
        
        # Attendance analysis
        attendance_rates = [s['attendance_rate'] for s in self.students]
        avg_attendance = sum(attendance_rates) / len(attendance_rates)
        print(f"Average Attendance Rate: {avg_attendance:.3f} ({avg_attendance*100:.1f}%)")
        
        # Store results
        self.analysis_results['basic_stats'] = {
            'total_students': total_students,
            'avg_gpa': avg_gpa,
            'min_gpa': min_gpa,
            'max_gpa': max_gpa,
            'performance_distribution': perf_counts,
            'avg_study_hours': avg_study_hours,
            'avg_attendance': avg_attendance
        }
    
    def correlation_analysis(self):
        """Simple correlation analysis."""
        if not self.students:
            print("No data available.")
            return
        
        print("\n" + "="*60)
        print("CORRELATION ANALYSIS")
        print("="*60)
        
        # Extract numerical features
        numerical_features = ['age', 'previous_gpa', 'study_hours_per_week', 
                             'attendance_rate', 'tech_comfort', 'stress_level', 'sleep_hours']
        
        # Calculate simple correlations with final GPA
        correlations = {}
        
        for feature in numerical_features:
            feature_values = [s[feature] for s in self.students]
            gpa_values = [s['final_gpa'] for s in self.students]
            
            # Simple correlation coefficient
            correlation = self.calculate_correlation(feature_values, gpa_values)
            correlations[feature] = correlation
        
        # Sort by absolute correlation
        sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        
        print("\nCorrelations with Final GPA (sorted by strength):")
        for feature, corr in sorted_corr:
            print(f"  {feature}: {corr:.3f}")
        
        self.analysis_results['correlations'] = correlations
    
    def calculate_correlation(self, x, y):
        """Calculate Pearson correlation coefficient."""
        n = len(x)
        if n == 0:
            return 0
        
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        
        sum_sq_x = sum((x[i] - mean_x) ** 2 for i in range(n))
        sum_sq_y = sum((y[i] - mean_y) ** 2 for i in range(n))
        
        denominator = math.sqrt(sum_sq_x * sum_sq_y)
        
        if denominator == 0:
            return 0
        
        return numerator / denominator
    
    def risk_assessment(self):
        """Identify students at risk."""
        if not self.students:
            print("No data available.")
            return
        
        print("\n" + "="*60)
        print("RISK ASSESSMENT")
        print("="*60)
        
        # Define risk criteria
        risk_factors = [
            ('Low Attendance', lambda s: s['attendance_rate'] < 0.8),
            ('Insufficient Study Time', lambda s: s['study_hours_per_week'] < 5),
            ('High Stress', lambda s: s['stress_level'] > 7),
            ('Insufficient Sleep', lambda s: s['sleep_hours'] < 6),
            ('Low Previous Performance', lambda s: s['previous_gpa'] < 2.5)
        ]
        
        # Calculate risk scores
        for student in self.students:
            risk_score = 0
            for factor_name, factor_func in risk_factors:
                if factor_func(student):
                    risk_score += 1
            student['risk_score'] = risk_score
            
            # Risk level
            if risk_score == 0:
                student['risk_level'] = 'Low Risk'
            elif risk_score <= 1:
                student['risk_level'] = 'Moderate Risk'
            elif risk_score <= 2:
                student['risk_level'] = 'High Risk'
            else:
                student['risk_level'] = 'Very High Risk'
        
        # Analyze risk distribution
        risk_distribution = {}
        for student in self.students:
            level = student['risk_level']
            risk_distribution[level] = risk_distribution.get(level, 0) + 1
        
        print("\nRisk Level Distribution:")
        for level, count in risk_distribution.items():
            percentage = (count / len(self.students)) * 100
            print(f"  {level}: {count} students ({percentage:.1f}%)")
        
        # High-risk students
        high_risk = [s for s in self.students if s['risk_score'] >= 3]
        print(f"\nHigh-risk students identified: {len(high_risk)}")
        
        self.analysis_results['risk_assessment'] = {
            'risk_distribution': risk_distribution,
            'high_risk_count': len(high_risk)
        }
    
    def simple_prediction_model(self):
        """Simple rule-based prediction model."""
        if not self.students:
            print("No data available.")
            return
        
        print("\n" + "="*60)
        print("SIMPLE PREDICTION MODEL")
        print("="*60)
        
        # Simple rule-based model
        correct_predictions = 0
        total_predictions = len(self.students)
        
        for student in self.students:
            # Predict based on key factors
            predicted_gpa = (
                student['previous_gpa'] * 0.4 +
                (student['study_hours_per_week'] / 10) * 1.0 +
                student['attendance_rate'] * 1.0 +
                (10 - student['stress_level']) / 10 * 0.5
            )
            
            # Predict category
            if predicted_gpa < 2.0:
                predicted_category = 'Poor'
            elif predicted_gpa < 3.0:
                predicted_category = 'Fair'
            elif predicted_gpa < 3.5:
                predicted_category = 'Good'
            else:
                predicted_category = 'Excellent'
            
            # Check if prediction is correct
            if predicted_category == student['performance_category']:
                correct_predictions += 1
        
        accuracy = (correct_predictions / total_predictions) * 100
        print(f"\nSimple Model Performance:")
        print(f"  Accuracy: {accuracy:.1f}% ({correct_predictions}/{total_predictions})")
        
        self.analysis_results['model_performance'] = {
            'accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'total_predictions': total_predictions
        }
    
    def predict_student_performance(self, student_profile):
        """Predict performance for a new student."""
        print("\n" + "="*60)
        print("STUDENT PERFORMANCE PREDICTION")
        print("="*60)
        
        # Simple prediction based on key factors
        predicted_gpa = (
            student_profile.get('previous_gpa', 3.0) * 0.4 +
            (student_profile.get('study_hours_per_week', 10) / 10) * 1.0 +
            student_profile.get('attendance_rate', 0.9) * 1.0 +
            (10 - student_profile.get('stress_level', 5)) / 10 * 0.5
        )
        
        # Predict category
        if predicted_gpa < 2.0:
            predicted_category = 'Poor'
        elif predicted_gpa < 3.0:
            predicted_category = 'Fair'
        elif predicted_gpa < 3.5:
            predicted_category = 'Good'
        else:
            predicted_category = 'Excellent'
        
        print(f"\nStudent Profile:")
        for key, value in student_profile.items():
            print(f"  {key}: {value}")
        
        print(f"\nPredictions:")
        print(f"  Predicted GPA: {predicted_gpa:.2f}")
        print(f"  Predicted Category: {predicted_category}")
        
        return predicted_gpa, predicted_category
    
    def generate_recommendations(self):
        """Generate actionable recommendations."""
        if not self.students or not self.analysis_results:
            print("No analysis results available.")
            return
        
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        
        correlations = self.analysis_results.get('correlations', {})
        
        print("\nKey Findings:")
        
        # Find strongest positive correlations
        strong_positive = [(k, v) for k, v in correlations.items() if v > 0.3]
        if strong_positive:
            print("  Strong positive factors:")
            for factor, corr in strong_positive:
                print(f"    - {factor} (r={corr:.3f})")
        
        # Find strongest negative correlations
        strong_negative = [(k, v) for k, v in correlations.items() if v < -0.3]
        if strong_negative:
            print("  Factors to address:")
            for factor, corr in strong_negative:
                print(f"    - {factor} (r={corr:.3f})")
        
        print("\nRecommendations for Institutions:")
        print("  1. Monitor and improve attendance rates")
        print("  2. Implement stress management programs")
        print("  3. Provide study skills workshops")
        print("  4. Ensure adequate technology support")
        print("  5. Early identification of at-risk students")
        
        print("\nRecommendations for Students:")
        print("  1. Maintain consistent attendance (>90%)")
        print("  2. Find optimal study hours balance")
        print("  3. Practice stress management techniques")
        print("  4. Ensure adequate sleep (7-9 hours)")
        print("  5. Seek help when struggling")
    
    def save_results(self, filename='analysis_results.json'):
        """Save analysis results to JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.analysis_results, f, indent=2)
        print(f"\nAnalysis results saved to {filename}")


def main():
    """Run the basic student performance analysis demo."""
    print("="*80)
    print("🎓 STUDENT PERFORMANCE ANALYSIS - BASIC DEMO")
    print("="*80)
    print("📅 Demo started at:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    # Initialize analyzer
    analyzer = SimpleStudentAnalyzer()
    
    # Step 1: Generate sample data
    print("\n📊 Step 1: Generating Sample Student Data...")
    students = analyzer.generate_sample_data(n_students=200)
    print(f"Generated data for {len(students)} students")
    
    # Step 2: Save data
    print("\n💾 Step 2: Saving Data...")
    analyzer.save_data_csv()
    
    # Step 3: Basic statistics
    print("\n📈 Step 3: Calculating Basic Statistics...")
    analyzer.basic_statistics()
    
    # Step 4: Correlation analysis
    print("\n🔍 Step 4: Performing Correlation Analysis...")
    analyzer.correlation_analysis()
    
    # Step 5: Risk assessment
    print("\n⚠️  Step 5: Conducting Risk Assessment...")
    analyzer.risk_assessment()
    
    # Step 6: Simple prediction model
    print("\n🤖 Step 6: Training Simple Prediction Model...")
    analyzer.simple_prediction_model()
    
    # Step 7: Example prediction
    print("\n🎯 Step 7: Example Student Prediction...")
    sample_student = {
        'age': 20,
        'previous_gpa': 3.2,
        'study_hours_per_week': 15.0,
        'attendance_rate': 0.9,
        'stress_level': 5,
        'sleep_hours': 7.5,
        'tech_comfort': 8
    }
    analyzer.predict_student_performance(sample_student)
    
    # Step 8: Generate recommendations
    print("\n💡 Step 8: Generating Recommendations...")
    analyzer.generate_recommendations()
    
    # Step 9: Save results
    print("\n💾 Step 9: Saving Results...")
    analyzer.save_results()
    
    print("\n" + "="*80)
    print("✅ BASIC DEMO COMPLETE!")
    print("="*80)
    print("📁 Generated Files:")
    print("  - demo_student_data.csv")
    print("  - analysis_results.json")
    print("\n🎯 This demo shows the basic structure of the ML system.")
    print("📊 For full functionality, install the complete dependencies.")


if __name__ == "__main__":
    main()