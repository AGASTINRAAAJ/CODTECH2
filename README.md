Name: Agastinraaj A

Company:CODTECH IT SOLUTIONS

ID:CTO8DS6242

Domain:Machine Learning

Duration:July to August 2024

Mentor:Muzammil Ahmed

Overview of the Script

The Python script is a machine learning workflow for sentiment analysis using a dataset of reviews. It utilizes a Naive Bayes classifier within a pipeline to classify the sentiment of text reviews.

Key Points

1. Libraries Imported
Essential libraries like pandas, numpy, sklearn, seaborn, and matplotlib are imported.

2. Dataset Loading and Exploration
The dataset review_data.csv is loaded using pandas.
Basic exploration is done by printing the first few rows and checking the distribution of sentiment labels.

3. Data Preprocessing
Review lengths are calculated, and reviews shorter than 20 characters are removed.
The sentiment labels are encoded into numerical values using LabelEncoder.

4. Splitting the Data
The dataset is split into features (X as the reviews) and target (y as the encoded sentiments).
The data is further split into training and testing sets using an 80-20 ratio.

5. Pipeline Definition
A pipeline is defined with two main components:
TF-IDF Vectorizer: Converts text data into numerical features using unigram and bigram tokenization.
Multinomial Naive Bayes Classifier: Used for classification of the reviews.

6. Cross-Validation
Cross-validation is performed to estimate the accuracy of the model on the training set.
The script prints the average cross-validation accuracy along with its standard deviation.

7. Hyperparameter Tuning
A grid search is conducted to optimize hyperparameters like n-gram range and smoothing parameter alpha for Naive Bayes using cross-validation.
The best parameters are printed.

8. Model Evaluation
The best model from the grid search is used to make predictions on the test set.
The accuracy of the model on the test set is printed.
A classification report detailing precision, recall, and F1-score for each sentiment class is generated.

9. Confusion Matrix Visualization
A confusion matrix is computed to visualize the performance of the model.
The confusion matrix is plotted using seaborn to show the true vs. predicted labels.

Conclusion

The script demonstrates a complete machine learning pipeline for sentiment analysis, from loading data to evaluating and visualizing the model's performance. The key aspects include data preprocessing, model training with cross-validation, hyperparameter tuning, and final model evaluation with a confusion matrix.
![IMG-20240831-WA0004](https://github.com/user-attachments/assets/6789a1b8-6040-4fb2-9b28-e8a0431bb4f9)
![IMG-20240831-WA0006](https://github.com/user-attachments/assets/136e5d98-6e4c-4d6f-9958-3a05fde5f84a)
