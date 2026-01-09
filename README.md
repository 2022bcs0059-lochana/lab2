## Lab 2: Automated ML Training with GitHub Actions

## Objective

This lab demonstrates the use of **GitHub Actions** to automate machine learning training, evaluation, and artifact storage, improving **reproducibility** and **experiment tracking**.

## Dataset

* **Wine Quality (Red Wine)**
* Source: UCI Machine Learning Repository
* Task: Regression (`quality` prediction)

## Workflow

On every **push or pull request to the main branch**, GitHub Actions:

* Sets up Python environment
* Installs dependencies
* Runs the training script
* Computes **MSE** and **R²**
* Displays metrics in Job Summary
* Uploads model and results as artifacts

## Experiments

Multiple experiments were performed by modifying the training script and committing each change separately.

* Linear Regression (baseline)
* Ridge Regression with feature selection
* Random Forest (50 trees)
* Random Forest (100 trees)

Each commit corresponds to one experiment and one CI run.

## Artifacts

Each run uploads:

* Trained model (`model.pkl`)
* Evaluation metrics (`metrics.json`)
