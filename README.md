<<<<<<< HEAD
# MLOps_RepoDemo
Repository for AxisBank Demo of Version control on Models
=======

# Loan Default Risk Scoring - GitHub Actions Setup

## Overview
This guide documents CI/CD setup for the PySpark ML model training pipeline.

## Prerequisites
- GitHub repository with `train.py`, `predict.py`, and `test_model.py`
- Training data CSV in repository or accessible location
- PySpark and dependencies configured

## Step-by-Step GitHub Actions Instructions

### 1. Create Workflow File
Create `.github/workflows/ml-pipeline.yml`:

```yaml
name: MLOps CI/CD Pipeline

on:
    push:
        branches: [main, develop]
    pull_request:
        branches: [main]

jobs:
    test:
        runs-on: ubuntu-latest
        steps:
            - uses: actions/checkout@v3
            
            - name: Set up Python
                uses: actions/setup-python@v4
                with:
                    python-version: '3.9'
            
            - name: Install dependencies
                run: |
                    pip install pyspark scikit-learn pytest
            
            - name: Run unit tests
                run: python -m pytest test_model.py -v

    train:
        needs: test
        runs-on: ubuntu-latest
        steps:
            - uses: actions/checkout@v3
            
            - name: Set up Python
                uses: actions/setup-python@v4
                with:
                    python-version: '3.9'
            
            - name: Install dependencies
                run: pip install pyspark
            
            - name: Train model
                run: python train.py --data_path ./data/training.csv --model_path ./models/loan_default_model
            
            - name: Upload model artifact
                uses: actions/upload-artifact@v3
                with:
                    name: trained-model
                    path: ./models/

    validate:
        needs: train
        runs-on: ubuntu-latest
        steps:
            - uses: actions/checkout@v3
            
            - name: Download model
                uses: actions/download-artifact@v3
            
            - name: Run predictions
                run: python predict.py --model_path ./trained-model
```

### 2. Configure Secrets (if needed)
Add sensitive data in repository Settings → Secrets:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`

### 3. Commit and Push
```bash
git add .github/workflows/ml-pipeline.yml
git commit -m "Add MLOps CI/CD pipeline"
git push origin main
```

### 4. Monitor Execution
View workflow runs in GitHub → Actions tab.
>>>>>>> a90a116 (Initial commit - Loan Risk MLOps project)
