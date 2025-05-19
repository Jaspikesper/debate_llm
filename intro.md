# Large Language Model Coding Project

Welcome to Jasper's LLM project! This project has 3 stages:

## 1. Building a foundational model

##### a. Tokenize and embed data

Tokenize the data into a usable sequence corpus for pretraining and fine-tuning.

##### b. Attention mechanism

Code the backbone of the transformer model: the self-attention blocks, which are crucial for adding contextual understanding across sequences.

##### c. Model architecture

High-level framework in Pytorch for running the large-language model, to be pretrained in the next step.

## 2. Pretraining the foundation model

##### a. Autoregressive training loop

Train the model in a self-supervised fashion. The task is to predict the next word in an ordered sequence.

##### b. Model evaluation and performance

Use a validation dataset to test the model's performance. The goal is avoiding overfitting while minimizing loss.

##### c. Store weights

## 3. Fine-tuning

##### a. As a text classifier

##### b. As a personal assistant
