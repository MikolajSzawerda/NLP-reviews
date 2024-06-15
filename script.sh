#!/bin/zsh
python -m venv venv
source venv/bin/activate

pip install jupyter torch torchmetrics transformers pandas lightning scikit-learn tensorboard