# Political Bias Detection in News Articles

## Overview

This project aims to develop a Chrome extension that detects political bias in news articles in real-time. By leveraging natural language processing (NLP) techniques, particularly BERT-based models, the system classifies articles into one of three categories: Left, Center, or Right. The Chrome extension extracts the text content of news articles from web pages and sends it to a model hosted on a server to predict the political bias of the article.

## Simple Workflow of Events

1. **User Browsing**: The user browses news articles on various websites.
2. **Text Extraction**: The Chrome extension automatically extracts the main content of the article from the webpage.
3. **Bias Prediction**: The extracted text is sent to the server, where the trained model performs a bias prediction (Left, Center, or Right).
4. **Display Results**: The Chrome extension displays the political bias classification (e.g., Left, Center, Right) on the article, possibly color-coded or as a notification.

---

## Tools and Technologies Used

- **Natural Language Processing (NLP)**: For analyzing and classifying the political bias in news articles.
- **BERT-based Model (RoBERTa)**: A pre-trained transformer-based model fine-tuned for political bias detection.
- **Python**: Programming language used for data preprocessing, model training, and server-side tasks.
- **Flask/FastAPI**: Backend frameworks for deploying the trained model as an API.
- **JavaScript**: For building the Chrome extension that communicates with the backend and interacts with the user interface.
- **HTML/CSS**: For designing the Chrome extension popup and UI elements.
- **PyTorch/TensorFlow**: Frameworks used for training the RoBERTa model.
- **GitHub**: Version control and hosting of the project files.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![BERT](https://img.shields.io/badge/BERT-FF5722?style=flat&logo=bert&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat&logo=javascript&logoColor=black)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)

---

## Datasets Used

1. **AllSides Ratings of Bias in Electronic Media**: This dataset provides a collection of articles labeled with political bias from various sources.
2. **Kaggle Dataset**: A dataset from Kaggle containing articles labeled as Left, Center, and Right, used for training and fine-tuning the model.
3. **Pranjali HuggingFace Dataset**: A curated dataset from HuggingFace for political bias detection, also used for fine-tuning and model validation.

---

## Data Preprocessing Steps

1. **Text Extraction**: Extract the content of articles from the respective datasets (AllSides, Kaggle, HuggingFace).
2. **Tokenization**: Split the text into tokens (words, subwords) using tokenizers like **HuggingFace's tokenizer**.
3. **Text Cleaning**: Remove stop words, special characters, and other unnecessary content to focus on the core meaning of the article.
4. **Normalization**: Convert all text to lowercase and handle any inconsistencies in the data.
5. **Data Augmentation**: Use techniques like SMOTE if needed for balancing the dataset in case of class imbalances.

**Libraries Used**:
- `pandas` (for data manipulation)
- `nltk`, `spaCy` (for text preprocessing and tokenization)
- `transformers` (HuggingFace's library for tokenization and model management)
- `scikit-learn` (for data splitting, cross-validation, etc.)

---

## Model Training

- **Model Used**: RoBERTa (a variant of BERT), which has been pre-trained and fine-tuned for the political bias detection task.
- **Training Process**:
  1. Fine-tune RoBERTa using the labeled dataset.
  2. Use the **Cross-Entropy Loss** function for multi-class classification.
  3. Evaluate the model using accuracy, precision, recall, and F1 score.
- **Libraries Used**:
  - `PyTorch`: For model building, training, and inference.
  - `transformers`: For managing pre-trained models like RoBERTa.

---

## Chrome Extension Details

The Chrome extension will perform the following tasks:

1. **Extract Article Content**: Using JavaScript, the extension extracts the article text from the webpage.
2. **Send to Server**: The extracted content is sent via an HTTP request to the server where the trained RoBERTa model is hosted.
3. **Display Result**: The extension receives the predicted political bias (Left, Center, Right) and displays it in the browser, either as a badge or in a popup.

**Key Features**:
- Real-time political bias detection while browsing.
- Clean and intuitive user interface (UI) for displaying results.

---

## Instructions to Run Locally

To run this project locally, follow these steps:

### 1. **Clone the Repository**

```bash
git clone https://github.com/peekayitachi/Political-Bias_test.git
cd Political-Bias_test
```
### 2. Set Up the Backend
Install Python dependencies:

```bash
pip install -r requirements.txt
#Train the model (if you haven't already). Otherwise, load the pre-trained model.
```
Start the FastAPI server:

```bash
python app.py
#This will start the backend API server at http://localhost:5000/.
```
### 3. Set Up the Chrome Extension

-Go to chrome://extensions/ in your Chrome browser.

-Enable Developer mode at the top right.

-Click on Load unpacked and select the extension folder from the cloned repository.

-Once installed, the extension will appear as an icon in your browser.

-Click on the extension icon when you're browsing news articles, and it will display the political bias prediction.

### 4. Running the Model
To test the model locally:

Input text or use a predefined article.

The Chrome extension will call the server's API endpoint and display the classification results.

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/peekayitachi/Political-Bias/blob/main/LICENSE) file for details.
