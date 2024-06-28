# HR Analytics and Employee Attrition Prediction

## Overview

This project explores HR analytics using a comprehensive dataset to understand employee demographics, employment status, performance, and other key metrics. Additionally, it builds and evaluates machine learning models to predict employee attrition.

## Dataset

The dataset used in this project is `HRDataset_v14.csv`. It contains various features such as employee information, job details, performance scores, and termination reasons.

## Exploratory Data Analysis (EDA)

The EDA section covers:
- Basic dataset overview with `head()`, `info()`, and `describe()`.
- Correlation heatmap to understand relationships between numerical features.
- Missing values analysis.
- Unique values and distribution analysis for categorical variables.
- Various visualizations using `seaborn` to understand the distribution and relationship between features.

## Data Preprocessing

- Handling of missing values.
- One-hot encoding for categorical variables.
- Feature scaling using `StandardScaler`.
- Splitting data into training and test sets.

## Machine Learning Models

Several models were trained and evaluated to predict employee attrition:
1. **Random Forest Classifier**
2. **Logistic Regression**
3. **Decision Tree Classifier**
4. **Support Vector Machine (SVM)**
5. **Naive Bayes Classifier**

### Model Evaluation

- Accuracy scores for each model.
- Confusion matrices visualized as heatmaps.
- Classification reports for precision, recall, and F1-score.
- ROC curves and AUC scores for performance comparison.

## Installation

To run this project, you'll need the following libraries:

```
pandas 
numpy 
seaborn 
matplotlib 
scikit-learn
```
You can just run the Jupyter Notebook or you can run this command seperately

```
!pip install -r requirements.txt
```

## Usage

1. Clone the repository:
   ```
   git clone https://github.com/Anirudh-Narra/HR_EDA-Project.git
   ```

2. Ensure you have the dataset (`HRDataset_v14.csv`) in the project directory.

3. Run the Jupyter Notebook to execute the analysis and models.

## Visualizations

- Correlation Heatmap
- Count plots for categorical variables (e.g., Marital Status, Department)
- Distribution plots for numerical variables (e.g., Salary)
- Heatmaps for confusion matrices of different models
- ROC curves for model performance comparison

## Conclusion

This project provides insights into employee demographics and performance using HR analytics. It also demonstrates the effectiveness of various machine learning models in predicting employee attrition.

## Future Work

- Enhance feature engineering to include more derived features.
- Experiment with advanced models like XGBoost or neural networks.
- Implement hyperparameter tuning for better model performance.
