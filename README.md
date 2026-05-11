AI-Driven Business Funding & Market Expansion Analytics Platform | Python, Scikit-learn, Pandas, Nashpy

• Developed an end-to-end predictive analytics and decision-support platform to analyze business funding potential, market expansion readiness, and operational performance using Machine Learning techniques.

• Built and evaluated multiple classification models including Random Forest, Gradient Boosting, and Support Vector Machine (SVM) for funding success prediction and business performance analysis.

• Implemented comprehensive data preprocessing pipelines involving data merging, feature engineering, missing value handling, standardization, categorical encoding, and market risk analysis.

• Engineered market readiness and foreign market risk indices using business, investor, and market trend data to support strategic decision-making.

• Applied hyperparameter tuning using GridSearchCV along with Stratified K-Fold Cross Validation to improve model robustness, generalization, and predictive accuracy.

• Generated comparative model evaluation reports using Accuracy, Precision, Recall, and F1-Score metrics.

• Developed analytical visualizations including market attractiveness heatmaps, model performance comparisons, and funding success distributions for business intelligence insights.

• Integrated investor preference analysis and strategic funding evaluation into a unified analytics workflow for startup funding assessment.


Code-


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nashpy as nash
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Set random seeds for reproducibility
np.random.seed(42)


class ForeignMarketExpansionIntelligence:

    def comprehensive_data_preprocessing(self, startup_data, market_trends, investor_data):
        # Merge startup and market trends data
        merged_data = pd.merge(
            startup_data,
            market_trends,
            on=['target_market', 'industry_sector'],
            how='left'
        )

        # Merge the investor data
        merged_data = pd.merge(
            merged_data,
            investor_data,
            left_on='startup_stage',
            right_on='investment_stage_preference',
            how='left'
        )

        # Calculate 'foreign_market_risk' based on market trends
        merged_data['foreign_market_risk'] = (
            merged_data['political_stability'] * 0.3
            + merged_data['economic_volatility'] * 0.4
            + merged_data['regulatory_complexity'] * 0.3
        )

        # Calculate market readiness index
        merged_data['market_readiness_index'] = (
            merged_data['technological_maturity'] * 0.4
            + merged_data['financial_stability'] * 0.3
            + merged_data['management_expertise'] * 0.3
        )

        # Derive binary funding success label
        merged_data['funding_success'] = (
            merged_data['market_readiness_index']
            > merged_data['market_readiness_index'].median()
        ).astype(int)

        return merged_data

    def advanced_ml_model_configuration(self):
        # Preprocessing pipeline for numerical and categorical data
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    'num',
                    Pipeline([
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler())
                    ]),
                    [
                        'market_readiness_index',
                        'financial_stability',
                        'foreign_market_risk',
                        'risk_tolerance',
                        'expected_return'
                    ]
                ),
                (
                    'cat',
                    OneHotEncoder(handle_unknown='ignore'),
                    ['industry_sector', 'target_market', 'startup_stage']
                )
            ]
        )

        # Define ML models with GridSearchCV
        ml_models = {
            'random_forest': Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', GridSearchCV(
                    RandomForestClassifier(random_state=42),
                    param_grid={
                        'n_estimators': [100, 200],
                        'max_depth': [5, 10, None],
                        'min_samples_split': [2, 5, 10]
                    },
                    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                ))
            ]),
            'gradient_boosting': Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', GridSearchCV(
                    GradientBoostingClassifier(random_state=42),
                    param_grid={
                        'n_estimators': [100, 200],
                        'learning_rate': [0.01, 0.1],
                        'max_depth': [3, 5]
                    },
                    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                ))
            ]),
            'support_vector_machine': Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', GridSearchCV(
                    SVC(probability=True, random_state=42),
                    param_grid={
                        'C': [0.1, 1, 10],
                        'kernel': ['rbf', 'linear']
                    },
                    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                ))
            ])
        }

        return ml_models

    def comprehensive_visualization(self, ml_results, preprocessed_data):
        plt.figure(figsize=(20, 7))

        # 1. ML Model Performance Comparison
        plt.subplot(131)
        model_performances = [result['accuracy'] for result in ml_results.values()]
        plt.bar(ml_results.keys(), model_performances)
        plt.title('Model Performance Comparison')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)

        # 2. Foreign Market Attractiveness Heatmap
        plt.subplot(132)
        market_heatmap = preprocessed_data.pivot_table(
            index='target_market',
            columns='industry_sector',
            values='market_readiness_index',
            aggfunc='mean'
        )
        sns.heatmap(market_heatmap, annot=True, cmap='YlGnBu')
        plt.title('Foreign Market Attractiveness')

        # 3. Funding Success Distribution
        plt.subplot(133)
        preprocessed_data['funding_success'].value_counts().plot(
            kind='pie', autopct='%1.1f%%'
        )
        plt.title('Funding Success Distribution')

        plt.tight_layout()
        plt.show()

    def generate_comprehensive_funding_report(self, ml_results):
        report = "Foreign Market Expansion Funding Intelligence Report\n"
        report += "==================================================\n\n"

        for model_name, result in ml_results.items():
            report += f"- {model_name.replace('_', ' ').title()}:\n"
            report += f"  Accuracy:  {result['accuracy']:.2%}\n"
            report += f"  Precision: {result['precision']:.2%}\n"
            report += f"  Recall:    {result['recall']:.2%}\n"
            report += f"  F1 Score:  {result['f1_score']:.2%}\n\n"

        return report


def main(startup_data, market_trends, investor_data):
    # Initialize Foreign Market Expansion Intelligence
    market_intelligence = ForeignMarketExpansionIntelligence()

    # Preprocess data
    preprocessed_data = market_intelligence.comprehensive_data_preprocessing(
        startup_data, market_trends, investor_data
    )

    # Configure ML models
    ml_models = market_intelligence.advanced_ml_model_configuration()
    ml_results = {}

    # Train and evaluate each model
    for name, model in ml_models.items():
        X = preprocessed_data.drop('funding_success', axis=1)
        y = preprocessed_data['funding_success']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        ml_results[name] = {
            'accuracy':  accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall':    recall_score(y_test, y_pred, zero_division=0),
            'f1_score':  f1_score(y_test, y_pred, zero_division=0)
        }

    # Visualize results
    market_intelligence.comprehensive_visualization(ml_results, preprocessed_data)

    # Print funding report
    funding_report = market_intelligence.generate_comprehensive_funding_report(ml_results)
    print(funding_report)


if __name__ == "__main__":
    # Load your datasets (replace paths as needed)
    startup_data = pd.read_csv("startup_data.csv")
    market_trends = pd.read_csv("market_trends.csv")
    investor_data = pd.read_csv("investor_data.csv")

    main(startup_data, market_trends, investor_data)
