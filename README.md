# Sentiment Analysis on Financial Text

This project aims to build a binary classifier that can distinguish between positive and non-positive (neutral and negative) sentiment in financial text using the [Llama-2 Finance Dataset](https://huggingface.co/datasets/AdiOO7/llama-2-finance/tree/main)

## Overview

The main objectives of this project are:

1. **Dataset**: Use the Llama-2 Finance Dataset, which contains text samples labeled with positive, neutral, and negative sentiment.
2. **Binary Classifier**: Implement a binary classifier using PyTorch to classify text as either positive or non-positive (neutral/negative).
3. **Evaluation Metrics**: Maximize the recall for the 'positive' class, ensuring that precision is not less than 85%. Additionally, report the specificity for the 'non-positive' class.
4. **Documentation**: Provide a detailed explanation of the approach, including data preprocessing, model selection, hyperparameter tuning, and evaluation.

Trained model final test metrics:

accuracy: 0.9505
precision: 0.9839
recall: 0.8478
f1: 0.9108

## Setup

To get started with this project, follow these steps:

1. **Clone the repository**:

git clone https://github.com/your-username/sentiment-analysis-finance.git

2. **Install dependencies**:

cd sentiment-analysis-finance
pip install -r requirements.txt

3. **Run the notebook**:

jupyter notebook sentimentAnalysis.ipynb

This will execute the sentiment analysis pipeline, including data preparation, model training, and evaluation.

## Project Structure

The project directory contains the following files:

- `sentimentAnalysis.ipynb`: The Jupyter Notebook that runs the sentiment analysis pipeline.
- `requirements.txt`: The list of dependencies required for the project.
- `README.md`: This file, which provides an overview and instructions for the project.

## Methodology

The project follows these steps:

1. **Data Preprocessing**: The text data from the Llama-2 Finance Dataset is cleaned and preprocessed, including tokenization, padding, and conversion to PyTorch tensors.

2. **Model Development**: The binary classifier is implemented using a simple PyTorch module. The model consists of a linear layer, a ReLU activation, and a final linear layer that outputs logits for the two classes (positive and non-positive).

3. **Training and Evaluation**: The model is trained using the PyTorch Lightning framework. The training objective is to minimize the cross-entropy loss between the model's predictions and the ground-truth labels. The model is trained for a fixed number of epochs, with the learning rate reduced on plateau of the validation accuracy metric.

The training parameters are set as follows:
- Batch size: 32
- Learning rate: 0.001
- Optimizer: AdamW
- Learning rate scheduler: ReduceLROnPlateau

This configuration was chosen based on common practices for text classification tasks and some initial experimentation to achieve good performance on the validation set.

4. **Documentation**: The `sentimentAnalysis.ipynb` notebook provides detailed comments and explanations for each step of the process, including data preprocessing, model architecture, training, and evaluation.

For more information, please refer to the comments and documentation within the `sentimentAnalysis.ipynb` notebook.

## Contributing

If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](https://opensource.org/license/mit)