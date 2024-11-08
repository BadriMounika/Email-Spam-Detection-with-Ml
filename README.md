# Email-spam-Detection-with-ML
<img src="emailspam.jpg" alt="Image description" width="800" height="500">>
# Project Title
### Email spam Detection with Machine Learning
# Description
This project implements a machine learning model to classify SMS messages as either spam or ham (not spam). The task utilizes Logistic Regression, a supervised learning algorithm, and evaluates its performance using multiple metrics, including accuracy, precision, recall, specificity, and confusion matrices. The project also includes a method for testing the classifier on new, unseen messages.

# Table of Contents
- [Project Overview](#Project-Overview)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Project Structure](#project-structure)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Evaluation Metrics](#evaluation-metrics)
- [Testing a Custom Message](#testing-a-custom-message)
- [Conclusion](#conclusion)
## Project Overview
The goal of this project is to develop a model that can identify spam messages from a collection of SMS messages. The dataset used in this project contains labeled messages, which are either spam or ham. The project demonstrates how to preprocess the text data, train a machine learning model (specifically Logistic Regression), and evaluate the model's performance.

By using TF-IDF (Term Frequency-Inverse Document Frequency) to vectorize text, we convert the raw message content into numerical data that can be processed by the model. The model is trained on 80% of the data, and 20% is held out for testing. The performance of the classifier is measured using various evaluation metrics like accuracy, precision, recall, and specificity.

## Dataset
The dataset used in this project is spam.csv, a collection of SMS messages labeled as either spam or ham. This dataset consists of two primary columns:

1.  **v1** : This column contains the category of the message, either 
   "spam" or "ham."
2. **v2** : This column contains the actual message text.



| v1    | v2                                                                 |
|-------|--------------------------------------------------------------------|
| ham   | Hey, how's it going?                                              |
| spam  | Congratulations! You've won a free iPhone!  
| ham   | Let's meet at the park tomorrow at 3 PM.
| spam  | WIN a free gift card! Call now!

The dataset contains irrelevant columns that need to be removed, such as extra unnamed columns. Additionally, any duplicate messages in the dataset are dropped to avoid bias.

## Dependencies
Before running the project, ensure the following Python libraries are installed:

- `numpy`: For handling numerical data.
- `pandas`: For data manipulation and analysis.
- `matplotlib`: For data visualization (used for plotting confusion matrix).
- `seaborn`: For advanced data visualization (used for plotting count plots).
- `scikit-learn`: For machine learning algorithms and evaluation tools.
- `nltk`: For natural language processing tasks like stopwords removal.
  
To install these libraries, use the following commands:

```sh
pip install numpy pandas matplotlib seaborn scikit-learn nltk
```

## Project Structure
The main steps involved in the project are:

1. **Data Loading**: Load the dataset into a Pandas DataFrame and examine its contents.
2. **Data Preprocessing**:
- Remove unnecessary columns (like unnamed ones).
- Remove duplicates to ensure data quality.
- Encode the target labels ('spam' and 'ham') into numeric values (0 for spam, 1 for ham).
3. **Text Vectorization**: Use TF-IDF Vectorizer to convert the text data into numerical format, which can be used by machine learning models.
4. **Model Training**: Split the data into training and testing sets (80%-20%) and train a Logistic Regression model.
5. **Model Evaluation**: Evaluate the model using accuracy, precision, recall, specificity, and a confusion matrix.
6. **Custom Message Testing**: Use the trained model to classify any custom message as spam or ham.
## Data Preprocessing
The dataset undergoes several preprocessing steps to prepare it for training:

1. **Removing Unnecessary Columns**: The dataset contains irrelevant columns (Unnamed: 2, Unnamed: 3, Unnamed: 4), which are dropped to focus on the useful data (`v1` and `v2`).
2. **Handling Duplicates**: Duplicate messages are dropped to prevent the model from being biased towards repeating data.
3. **Label Encoding**: The target variable (`v1`) is encoded as numeric labels where:
- "spam" is converted to `0`
- "ham" is converted to `1`
4. **Text Vectorization**: **TF-IDF** (Term Frequency-Inverse Document Frequency) is used to convert the raw text into a numerical matrix. This technique helps capture the importance of words relative to the entire corpus and reduces the influence of frequently occurring but less informative words (e.g., "the", "is", etc.).
## Model Training
The dataset is split into two parts: one for training the model and the other for testing its performance.

1. **Train-Test Split**: The data is split into 80% training data and 20% testing data using `train_test_split` from scikit-learn.
2. **Logistic Regression**: Logistic Regression is chosen for this binary classification task. It is a simple yet effective model for classification problems.
3. **Model Fitting**: The model is trained using the fit method on the training data.
## Evaluation Metrics
After training the model, we evaluate its performance using several key metrics:

1. **Accuracy**: The percentage of correct predictions made by the model.
2. **Confusion Matrix**: A confusion matrix is plotted to visualize the performance of the classification model. It shows the true positive, true negative, false positive, and false negative values.
3. **Precision**: Precision measures how many of the messages predicted as spam were actually spam. It is calculated as:
- `Precision = TP / (TP + FP)`
4. **Recall**: Recall measures how many of the actual spam messages were correctly identified. It is calculated as:
- `Recall = TP / (TP + FN)`
5. **Specificity**: Specificity measures how well the model avoids classifying ham messages as spam. It is calculated as:
- `Specificity = TN / (TN + FP)`
## Testing a Custom Message
Once the model is trained, it can be used to classify new, unseen messages. The following code snippet demonstrates how to classify a custom message:

```sh
input_your_mail = "Congratulations! You've won a luxury vacation package worth $10,000!"
input_data_features = feature_extraction.transform([input_your_mail])
prediction = model.predict(input_data_features)
print("Ham Mail" if prediction[0] == 1 else "Spam Mail")
```
This will print either `"Spam Mail"` or `"Ham Mail"` based on the classification result of the input message.

## Conclusion
This SMS Spam Detection project demonstrates the use of Logistic Regression for classifying text data. By preprocessing the data with TF-IDF and training a logistic regression model, the project achieves a high level of accuracy and is capable of making real-time predictions on new messages. The performance metrics provide a comprehensive evaluation of how well the model is classifying spam messages and distinguishing them from non-spam messages.


