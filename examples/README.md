# AIVM Examples

This directory contains a series of example Jupyter notebooks and pre-trained model files that demonstrate how to use AIVM for privacy-preserving inference, fine-tuning, and custom model deployment. Below is a brief overview of each example:

## Notebooks

### 1. `1-getting-started.ipynb` 
This notebook is your introduction to AIVM. It walks you through setting up the AIVM devnet and performing a basic private inference task using a pre-trained model (e.g., LeNet5 for MNIST data).

### 2. `2a-fine-tuning-bert-tiny.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NillionNetwork/aivm/blob/main/examples/2a-fine-tuning-bert-tiny.ipynb)
This example demonstrates how to fine-tune the BertTiny model on a new dataset. It includes steps for loading the pre-trained BertTiny model and fine-tuning it on a different classification task.

### 3. `2b-fine-tuning-lenet5.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NillionNetwork/aivm/blob/main/examples/2b-fine-tuning-lenet5.ipynb)
Similar to `2a`, this notebook covers the fine-tuning process for the LeNet5 model, which is designed for image classification tasks. The example uses a sample dataset to show how to fine-tune LeNet5 to new image data.

### 4. `2c-fine-tuning-bert-tiny-tweet-dataset.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NillionNetwork/aivm/blob/main/examples/2c-fine-tuning-bert-tiny-tweet-dataset.ipynb)
This notebook extends the fine-tuning process specifically for **BertTiny** using a tweet sentiment classification dataset. It showcases how to adapt the model for a real-world text-based task (positive, neutral, negative sentiment classification).

### 5. `3a-upload-your-bert-tiny-model.ipynb`
In this notebook, you’ll learn how to upload a custom **BertTiny** model to AIVM. It covers how to prepare your model, encrypt the input, and upload it to AIVM for secure inference.

### 6. `3b-upload-your-lenet5-model.ipynb`
This example explains how to upload a custom **LeNet5** model to AIVM. The notebook provides step-by-step instructions to deploy your trained model for privacy-preserving inference.

### 7. `3c-upload-your-bert-tiny-for-tweet-sentiment.ipynb`
This notebook focuses on deploying a **BertTiny** model that has been fine-tuned for tweet sentiment classification. It demonstrates how to upload the model and perform secure sentiment analysis on tweet data.

## Model Files

### 1. `cats_dogs_lenet5.pth`
A pre-trained **LeNet5** model designed for the Cats vs. Dogs dataset. This file can be used in the fine-tuning and upload notebooks for LeNet5.

### 2. `imdb_bert_tiny.onnx`
A pre-trained **BertTiny** model for sentiment classification on the IMDB dataset. This file can be used for both fine-tuning and deploying custom models using AIVM.

### 3. `twitter_bert_tiny.onnx`
A pre-trained **BertTiny** model for sentiment classification on Twitter data. This file is specifically used in notebooks related to tweet sentiment classification and can be uploaded to AIVM for inference.

---

Each notebook is designed to be run sequentially or independently based on your specific use case. Refer to the main [README](../README.md) for more information on how to set up and run AIVM.