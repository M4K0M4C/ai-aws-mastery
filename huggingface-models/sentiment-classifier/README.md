# Hugging Face Sentiment Classifier

This project fine-tunes a transformer model (DistilBERT) on a sentiment classification task.

## Contents
- `train.py`: Fine-tunes the model
- `evaluate.py`: Evaluates performance
- `deploy_sagemaker_stub.py`: Placeholder for deployment logic
- `sentiment_classifier.ipynb`: Jupyter notebook version
- `data/sample.csv`: Toy dataset

## Requirements
- `transformers`, `datasets`, `torch`, `scikit-learn`
- AWS CLI & Boto3 (for deployment)

## Instructions
1. Run `train.py` to fine-tune model
2. Run `evaluate.py` to compute accuracy
3. Modify and use `deploy_sagemaker_stub.py` for cloud deployment
