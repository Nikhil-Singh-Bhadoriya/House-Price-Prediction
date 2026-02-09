# House Price Prediction - Advanced Regression Techniques

A comprehensive machine learning project for predicting house prices using advanced regression techniques and neural networks. This project implements a complete data science pipeline from data analysis to model deployment.

## 📊 Project Overview

The main objective of this project is to predict house sale prices based on various features such as location, size, quality, and amenities. The project demonstrates a professional approach to machine learning by following industry best practices and covering all phases of a data science lifecycle.

## 🎯 Dataset

The dataset is sourced from the Kaggle competition: [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)

**Dataset Shape:** 1460 rows with multiple features

## 🔄 Data Science Lifecycle

This project follows a structured approach covering all essential phases:

1. **Data Analysis**
2. **Feature Engineering**
3. **Feature Selection**
4. **Model Building**
5. **Model Deployment**

---

## 📈 Phase 1: Exploratory Data Analysis

### Objectives
The EDA phase focuses on understanding the data thoroughly by analyzing:

- **Missing Values**: Identification and analysis of missing data patterns
- **Outliers**: Detection of anomalous data points
- **Numerical Variables**: Analysis and distribution of continuous features
- **Categorical Variables**: Examination of discrete features and their cardinality
- **Feature Relationships**: Correlation analysis between independent features and the target variable (SalePrice)

### Key Findings

#### Missing Values Analysis
- Identified features with missing values and calculated their percentages
- Key features with significant missing data:
  - **PoolQC**: High percentage of missing values
  - **MiscFeature**: High percentage of missing values
  - **Alley**: ~93.8% missing values (1369/1460)
  - **Fence**: Substantial missing values
  - **FireplaceQu**: Notable missing values
  - **LotFrontage**: Numerical feature with missing values
  - **GarageType, GarageFinish, GarageQual, GarageCond**: Garage-related features with missing values
  - **Basement-related features**: Multiple basement features with missing values

#### Visualizations
- Created visualizations to understand the relationship between missing values and SalePrice
- Distribution plots for numerical variables
- Categorical feature analysis through count plots and box plots

### Technologies Used
```python
- pandas
- numpy
- matplotlib
- seaborn
```

---

## 🔧 Phase 2: Feature Engineering

### Objectives
Transform raw data into features that better represent the underlying problem to improve model performance.

### Feature Engineering Steps

#### 1. **Data Splitting**
- Split data into training and test sets (90-10 split)
- **Training Set**: 1314 samples
- **Test Set**: 146 samples
- Used `train_test_split` with `random_state=0` to prevent data leakage

#### 2. **Handling Missing Values**

**Categorical Features:**
- Replaced missing values with a new label: `'Missing'`
- This approach treats the absence of data as a separate category
- Applied to features like Alley, FireplaceQu, Fence, PoolQC, etc.

**Numerical Features:**
- Filled missing values with the **median** of the feature
- Created additional binary indicator features (e.g., `LotFrontage_nan`) to capture whether the value was missing
- This preserves information about missingness which can be predictive

#### 3. **Temporal Variables**
- Analyzed and processed time-related features
- Created new features based on temporal relationships

#### 4. **Categorical Variable Processing**
- Removed rare labels in categorical variables
- Reduced cardinality to prevent overfitting
- Applied appropriate encoding techniques

#### 5. **Feature Standardization**
- Standardized variable values to the same range
- Ensured all features contribute equally to model training

### Key Transformations
- Categorical features: Filled missing with 'Missing' label
- Numerical features: Median imputation + missingness indicators
- Feature scaling and standardization applied

---

## 🎯 Phase 3: Feature Selection and Model Creation

### Feature Selection Strategy

#### Lasso Regression for Feature Selection
- Used **Lasso (L1 Regularization)** with `alpha=0.005`
- Employed `SelectFromModel` from sklearn to automatically select features
- **Selection Criteria**: Features with non-zero coefficients retained

#### Results
- **Total Features**: Originally had numerous features after engineering
- **Selected Features**: Reduced to a subset of most predictive features
- **Features Eliminated**: Features with coefficients shrunk to zero by Lasso

This process reduces dimensionality, prevents overfitting, and improves model interpretability.

---

## 🤖 Models Implemented

### 1. XGBoost Regressor

**XGBoost (Extreme Gradient Boosting)** is an ensemble learning method that builds multiple decision trees sequentially.

#### Configuration
```python
import xgboost
regressor = xgboost.XGBRegressor()
```

#### Features
- Gradient boosting framework
- Handles missing values automatically
- Built-in regularization to prevent overfitting
- Fast and efficient training

#### Output
- Generated predictions on test dataset
- Created submission file: `sample_submission.csv`

---

### 2. Artificial Neural Network (ANN)

Built a **deep learning model** using Keras with PyTorch backend.

#### Architecture
```
Input Layer → Dense(50, ReLU) → Dense(25, ReLU) → Dense(50, ReLU) → Output(1)
```

#### Model Details
- **Input Layer**: 50 neurons with ReLU activation
- **Hidden Layers**: 
  - Layer 1: 25 neurons with ReLU activation
  - Layer 2: 50 neurons with ReLU activation
- **Output Layer**: 1 neuron (continuous price prediction)
- **Initialization**: He uniform initialization
- **Optimizer**: Adamax
- **Loss Function**: Root Mean Squared Error (RMSE)

#### Training Configuration
- **Epochs**: 151
- **Batch Size**: 10
- **Validation Split**: 20%
- **Backend**: PyTorch (Keras 3 compatible)

#### Custom Loss Function
```python
def root_mean_squared_error(y_true, y_pred):
    return ops.sqrt(ops.mean(ops.square(y_pred - y_true)))
```

#### Output
- Model training history tracked
- Predictions generated on test data
- Final submission file created

---

## 📁 Project Structure

```
House-Price-Prediction-main/
│
├── Exploratory Data Analysis.ipynb      # Phase 1: Data analysis and visualization
├── Feature Engineering.ipynb            # Phase 2: Feature transformations
├── Feature Selection and model creation.ipynb  # Phase 3: Model building
├── train.csv                           # Training dataset (to be downloaded)
├── test.csv                            # Test dataset (to be downloaded)
└── sample_submission.csv               # Model predictions output
```

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost keras torch
```

### Installation & Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd House-Price-Prediction-main
```

2. **Download the dataset**
   - Visit [Kaggle Competition Page](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
   - Download `train.csv` and `test.csv`
   - Place them in the project root directory

3. **Run the notebooks in sequence**
   - Start with `Exploratory Data Analysis.ipynb`
   - Then run `Feature Engineering.ipynb`
   - Finally execute `Feature Selection and model creation.ipynb`

---

## 📊 Results

### Model Performance

Both XGBoost and ANN models were trained and generated predictions:

- **XGBoost**: Efficient gradient boosting with automatic feature handling
- **ANN**: Deep learning approach with 151 epochs of training
- **Output**: Predictions saved in `sample_submission.csv`

### Submission Format
The final predictions are formatted for Kaggle submission with columns:
- `Id`: Property identifier
- `SalePrice`: Predicted house price

---

## 🛠️ Technical Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.x |
| **Data Processing** | pandas, numpy |
| **Visualization** | matplotlib, seaborn |
| **Machine Learning** | scikit-learn, XGBoost |
| **Deep Learning** | Keras 3 with PyTorch backend |
| **Feature Selection** | Lasso Regression (L1) |

---

## 📝 Key Learnings

1. **Data Preprocessing is Critical**: Proper handling of missing values significantly impacts model performance
2. **Feature Engineering**: Creating meaningful features and indicators for missingness improves predictions
3. **Feature Selection**: Using Lasso helps identify the most important features and reduces model complexity
4. **Model Diversity**: Comparing traditional ML (XGBoost) with deep learning (ANN) provides insights into different approaches
5. **Pipeline Approach**: Following a structured data science lifecycle ensures reproducibility and maintainability

