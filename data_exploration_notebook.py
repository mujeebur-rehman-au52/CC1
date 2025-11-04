"""
Student Performance Data Exploration and Advanced Analytics

This module provides advanced data exploration, statistical analysis, and visualization
for student performance data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class StudentDataExplorer:
    """
    Advanced data exploration and analytics for student performance data.
    """
    
    def __init__(self, data_path=None):
        self.data = None
        self.numerical_features = []
        self.categorical_features = []
        
        if data_path:
            self.load_data(data_path)
    
    def load_data(self, data_path):
        """Load student data from CSV file."""
        try:
            self.data = pd.read_csv(data_path)
            self._identify_feature_types()
            print(f"Data loaded successfully: {self.data.shape[0]} students, {self.data.shape[1]} features")
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def _identify_feature_types(self):
        """Identify numerical and categorical features."""
        if self.data is not None:
            self.numerical_features = self.data.select_dtypes(include=[np.number]).columns.tolist()
            self.categorical_features = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Remove ID columns from analysis
            if 'student_id' in self.numerical_features:
                self.numerical_features.remove('student_id')
    
    def basic_statistics(self):
        """Generate basic statistical summary of the data."""
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
        
        print("="*70)
        print("BASIC STATISTICAL SUMMARY")
        print("="*70)
        
        print("\nDataset Shape:", self.data.shape)
        print("\nData Types:")
        print(self.data.dtypes)
        
        print("\nMissing Values:")
        missing_data = self.data.isnull().sum()
        if missing_data.sum() == 0:
            print("No missing values found!")
        else:
            print(missing_data[missing_data > 0])
        
        print("\nNumerical Features Summary:")
        print(self.data[self.numerical_features].describe())
        
        print("\nCategorical Features Summary:")
        for feature in self.categorical_features:
            if feature != 'student_id':
                print(f"\n{feature}:")
                print(self.data[feature].value_counts())
    
    def correlation_analysis(self):
        """Perform correlation analysis on numerical features."""
        if self.data is None:
            return
        
        # Calculate correlation matrix
        corr_matrix = self.data[self.numerical_features].corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Correlation Matrix of Numerical Features')
        plt.tight_layout()
        plt.savefig('/workspace/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Find strong correlations
        print("\nStrong Correlations (|r| > 0.5) with Final GPA:")
        if 'final_gpa' in self.numerical_features:
            gpa_corr = corr_matrix['final_gpa'].abs().sort_values(ascending=False)
            strong_corr = gpa_corr[gpa_corr > 0.5]
            for feature, corr in strong_corr.items():
                if feature != 'final_gpa':
                    print(f"{feature}: {corr:.3f}")
    
    def distribution_analysis(self):
        """Analyze distributions of key variables."""
        if self.data is None:
            return
        
        # Create distribution plots
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Distribution Analysis of Key Variables', fontsize=16)
        
        key_vars = ['final_gpa', 'previous_gpa', 'study_hours_per_week', 'attendance_rate',
                   'stress_level', 'sleep_hours', 'tech_comfort', 'age']
        
        for i, var in enumerate(key_vars[:8]):
            row = i // 3
            col = i % 3
            
            if var in self.data.columns:
                axes[row, col].hist(self.data[var], bins=20, alpha=0.7, edgecolor='black')
                axes[row, col].set_title(f'Distribution of {var}')
                axes[row, col].set_xlabel(var)
                axes[row, col].set_ylabel('Frequency')
                
                # Add statistics
                mean_val = self.data[var].mean()
                std_val = self.data[var].std()
                axes[row, col].axvline(mean_val, color='red', linestyle='--', 
                                     label=f'Mean: {mean_val:.2f}')
                axes[row, col].legend()
        
        # Remove empty subplot
        if len(key_vars) < 9:
            fig.delaxes(axes[2, 2])
        
        plt.tight_layout()
        plt.savefig('/workspace/distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def performance_segmentation(self):
        """Segment students based on performance and analyze characteristics."""
        if self.data is None or 'final_gpa' not in self.data.columns:
            return
        
        # Create performance segments
        self.data['performance_segment'] = pd.cut(self.data['final_gpa'], 
                                                bins=[0, 2.5, 3.0, 3.5, 4.0],
                                                labels=['Low', 'Average', 'Good', 'Excellent'])
        
        # Analyze segments
        print("Performance Segmentation Analysis:")
        print("="*50)
        
        segment_stats = self.data.groupby('performance_segment')[self.numerical_features].mean()
        print("\nAverage characteristics by performance segment:")
        print(segment_stats.round(2))
        
        # Visualize segment characteristics
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Student Characteristics by Performance Segment', fontsize=16)
        
        key_features = ['study_hours_per_week', 'attendance_rate', 'stress_level', 
                       'sleep_hours', 'tech_comfort', 'previous_gpa']
        
        for i, feature in enumerate(key_features):
            row = i // 3
            col = i % 3
            
            if feature in self.data.columns:
                sns.boxplot(data=self.data, x='performance_segment', y=feature, ax=axes[row, col])
                axes[row, col].set_title(f'{feature} by Performance Segment')
                axes[row, col].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('/workspace/performance_segmentation.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def cluster_analysis(self):
        """Perform K-means clustering to identify student profiles."""
        if self.data is None:
            return
        
        # Prepare data for clustering
        cluster_features = ['study_hours_per_week', 'attendance_rate', 'stress_level', 
                           'sleep_hours', 'tech_comfort', 'previous_gpa']
        
        cluster_data = self.data[cluster_features].dropna()
        
        # Standardize features
        scaler = StandardScaler()
        cluster_data_scaled = scaler.fit_transform(cluster_data)
        
        # Determine optimal number of clusters using elbow method
        inertias = []
        k_range = range(2, 11)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(cluster_data_scaled)
            inertias.append(kmeans.inertia_)
        
        # Plot elbow curve
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, inertias, 'bo-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method for Optimal k')
        plt.grid(True)
        plt.savefig('/workspace/elbow_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Perform clustering with optimal k (assuming k=4)
        optimal_k = 4
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        cluster_labels = kmeans.fit_predict(cluster_data_scaled)
        
        # Add cluster labels to data
        cluster_data['cluster'] = cluster_labels
        
        # Analyze clusters
        print("Cluster Analysis Results:")
        print("="*40)
        
        cluster_summary = cluster_data.groupby('cluster').mean()
        print("\nCluster characteristics (means):")
        print(cluster_summary.round(2))
        
        # Visualize clusters using PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(cluster_data_scaled)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=cluster_labels, 
                             cmap='viridis', alpha=0.6)
        plt.xlabel(f'First Principal Component (explains {pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'Second Principal Component (explains {pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('Student Clusters (PCA Visualization)')
        plt.colorbar(scatter)
        plt.savefig('/workspace/cluster_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return cluster_labels
    
    def risk_assessment(self):
        """Identify students at risk of poor performance."""
        if self.data is None:
            return
        
        # Define risk factors (adjust thresholds as needed)
        risk_conditions = [
            (self.data['attendance_rate'] < 0.8, 'Low Attendance'),
            (self.data['study_hours_per_week'] < 5, 'Insufficient Study Time'),
            (self.data['stress_level'] > 7, 'High Stress'),
            (self.data['sleep_hours'] < 6, 'Insufficient Sleep'),
            (self.data['previous_gpa'] < 2.5, 'Low Previous Performance')
        ]
        
        # Calculate risk scores
        self.data['risk_score'] = 0
        risk_details = []
        
        for condition, description in risk_conditions:
            self.data['risk_score'] += condition.astype(int)
            risk_count = condition.sum()
            risk_details.append((description, risk_count, risk_count/len(self.data)*100))
        
        # Categorize risk levels
        self.data['risk_level'] = pd.cut(self.data['risk_score'], 
                                       bins=[-1, 0, 1, 2, 5],
                                       labels=['Low Risk', 'Moderate Risk', 'High Risk', 'Very High Risk'])
        
        print("Risk Assessment Analysis:")
        print("="*40)
        
        print("\nRisk Factor Distribution:")
        for description, count, percentage in risk_details:
            print(f"{description}: {count} students ({percentage:.1f}%)")
        
        print("\nRisk Level Distribution:")
        risk_distribution = self.data['risk_level'].value_counts()
        for level, count in risk_distribution.items():
            percentage = count / len(self.data) * 100
            print(f"{level}: {count} students ({percentage:.1f}%)")
        
        # Visualize risk assessment
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Risk score distribution
        ax1.hist(self.data['risk_score'], bins=range(6), alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Risk Score')
        ax1.set_ylabel('Number of Students')
        ax1.set_title('Distribution of Risk Scores')
        
        # Risk level vs GPA
        sns.boxplot(data=self.data, x='risk_level', y='final_gpa', ax=ax2)
        ax2.set_title('Final GPA by Risk Level')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('/workspace/risk_assessment.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Identify high-risk students
        high_risk_students = self.data[self.data['risk_level'].isin(['High Risk', 'Very High Risk'])]
        print(f"\nHigh-risk students identified: {len(high_risk_students)} students")
        
        return high_risk_students
    
    def interactive_dashboard_data(self):
        """Prepare data for interactive dashboard visualization."""
        if self.data is None:
            return None
        
        # Create interactive plots using Plotly
        
        # 1. 3D Scatter plot
        fig_3d = px.scatter_3d(self.data, x='study_hours_per_week', y='attendance_rate', 
                              z='final_gpa', color='performance_category',
                              title='3D Student Performance Analysis',
                              hover_data=['age', 'stress_level', 'sleep_hours'])
        fig_3d.write_html('/workspace/interactive_3d_plot.html')
        
        # 2. Correlation heatmap
        corr_matrix = self.data[self.numerical_features].corr()
        fig_heatmap = px.imshow(corr_matrix, 
                               title='Interactive Correlation Heatmap',
                               color_continuous_scale='RdBu')
        fig_heatmap.write_html('/workspace/interactive_correlation.html')
        
        # 3. Performance dashboard
        fig_dashboard = make_subplots(
            rows=2, cols=2,
            subplot_titles=['GPA Distribution', 'Study Hours vs GPA', 
                           'Stress Level Impact', 'Performance by Gender'],
            specs=[[{"type": "histogram"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Add traces
        fig_dashboard.add_histogram(x=self.data['final_gpa'], name='GPA Distribution', row=1, col=1)
        fig_dashboard.add_scatter(x=self.data['study_hours_per_week'], y=self.data['final_gpa'], 
                                mode='markers', name='Study vs GPA', row=1, col=2)
        
        stress_gpa = self.data.groupby('stress_level')['final_gpa'].mean()
        fig_dashboard.add_bar(x=stress_gpa.index, y=stress_gpa.values, 
                            name='Stress Impact', row=2, col=1)
        
        gender_gpa = self.data.groupby('gender')['final_gpa'].mean()
        fig_dashboard.add_bar(x=gender_gpa.index, y=gender_gpa.values, 
                            name='Gender Performance', row=2, col=2)
        
        fig_dashboard.update_layout(title_text="Student Performance Interactive Dashboard", 
                                  showlegend=False)
        fig_dashboard.write_html('/workspace/interactive_dashboard.html')
        
        print("Interactive visualizations saved as HTML files:")
        print("- interactive_3d_plot.html")
        print("- interactive_correlation.html") 
        print("- interactive_dashboard.html")


def main():
    """Main function to run the data exploration analysis."""
    print("Starting Student Performance Data Exploration...")
    
    # Initialize explorer
    explorer = StudentDataExplorer()
    
    # Check if data file exists, if not, generate it
    try:
        explorer.load_data('/workspace/student_data.csv')
    except:
        print("Data file not found. Generating sample data...")
        from student_performance_analyzer import StudentPerformanceAnalyzer
        analyzer = StudentPerformanceAnalyzer()
        sample_data = analyzer.generate_sample_data(1000)
        sample_data.to_csv('/workspace/student_data.csv', index=False)
        explorer.load_data('/workspace/student_data.csv')
    
    # Run all analyses
    print("\n1. Basic Statistics Analysis...")
    explorer.basic_statistics()
    
    print("\n2. Correlation Analysis...")
    explorer.correlation_analysis()
    
    print("\n3. Distribution Analysis...")
    explorer.distribution_analysis()
    
    print("\n4. Performance Segmentation...")
    explorer.performance_segmentation()
    
    print("\n5. Cluster Analysis...")
    explorer.cluster_analysis()
    
    print("\n6. Risk Assessment...")
    explorer.risk_assessment()
    
    print("\n7. Creating Interactive Visualizations...")
    explorer.interactive_dashboard_data()
    
    print("\nData exploration complete! All visualizations and analyses have been saved.")


if __name__ == "__main__":
    main()