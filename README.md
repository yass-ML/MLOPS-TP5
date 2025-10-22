# ML Model Transpiler

Convert scikit-learn models to C code for deployment.

## Setup

```bash
# Create virtual environment
python -m venv <env_name>
source <env_name>/bin/activate  # On Windows: <env_name>\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Train Models

```bash
cd data
python train_model.py          # Linear regression
python train_logistic_model.py # Logistic regression
python train_tree_clf.py       # Decision tree classifier
python train_tree_reg.py       # Decision tree regressor
```

### 2. Test Transpiler

Open `transpilation_tests.ipynb` to:
- Transpile models to C code
- Compile and run C implementations
- Compare Python vs C predictions
- View accuracy/R² scores

## Supported Models

- Linear Regression
- Logistic Regression (binary)
- Decision Tree Classifier (binary)
- Decision Tree Regressor
