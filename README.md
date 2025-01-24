# Natural Language Processing (NLP) Project: Sentiment Analysis on Tweets about Apple and Google Products

![](https://github.com/mwangikelvin201/GROUP7-Phase4/blob/16fa23d6bb000927274868998d8dd1be3f3de10e/360_F_411509944_NHQwlYfg1td6fBQyyHLdlfltmlv8cmAp.jpg)
# Overview
This project aims to analyze Twitter sentiment about Apple and Google products using Natural Language Processing (NLP). The dataset contains tweets labeled as positive, negative, or neutral. By building a sentiment analysis model, we aim to categorize the sentiment of tweets accurately and gain insights into public perception of these tech giants' products.

# Business Understanding
## Business Problem
Understanding customer sentiment is critical for businesses to gauge public opinion and improve products or services. For Apple and Google, analyzing Twitter sentiment can provide actionable insights to enhance user satisfaction and market strategies.

## Stakeholders
### Marketing Teams:
Use insights to create targeted campaigns focusing on products with positive sentiment.
Address negative feedback to improve brand perception.

### Product Teams:
Identify areas of improvement for specific products (e.g., iPhone or Pixel).

### Executives:
Make data-driven decisions for product launches, pricing strategies, and market positioning.

# Objectives
Build a model to classify the sentiment of tweets into positive, negative, or neutral categories.
Evaluate model performance using suitable metrics.
Provide insights and recommendations based on the analysis results.

## Requirements to run the project
Python 3.x

scikit-learn
 
nltk

## How to access the project for any contributions(Open to contributions)

git clone https://github.com/mwangikelvin201/GROUP7-Phase4

# Data Understanding
Here, we intend to comprehensively explore and analyze our dataset to gain insights into its structure, content, and quality.

We look at different visualizations from our dataset to gain the above understanding

## countplot of brands

![](![image](https://github.com/mwangikelvin201/GROUP7-Phase4/blob/bacd04822f2cfec71164e8e21de7fce784d7f4a8/image2.png)

## sentiment distribution by brand 
![](https://github.com/mwangikelvin201/GROUP7-Phase4/blob/bacd04822f2cfec71164e8e21de7fce784d7f4a8/image3.png)

## Data Preprocessing

Here the data is put through various processes below to prepare it for modelling

##  Tokenization 
splits text into individual words
  
## Stopword removal
removes common words like "the", "is", "at"
  
## Lemmatization 
reduces words to their base form

# Modelling and evaluation

The data is put through different models to check its performance.

## Models  used:

### 1. Logistic Regression

In the first case, we built a base model using Logistic Regression for multinomial classification (positive, negative, and neutral sentiments). Using any of the different vectorizers, we can see that the model has very poor recall score. This is due to the class imbalance in the dataset.

In the second Logistic Regression model we use the balanced class weights, we can see that the model recall values increase but the precision values decrease. This is due to the fact that the model is now predicting more false positives.

### 2. Naive Bayes Model

The model has improved on accuracy score to the Logistic Regression model but the recall score is still very low especially for the positive sentiment.The model seems to biased towards the neutral sentiment.

### 3. Random Forest, count vectorizer,SMOTE

With an accuracy of 97, Random Forest model has improved on the recall score having recall above 90 for all the sentiments compared to the Naive Bayes model and has better accuracy scores than both the Logistic Regression and Naive Bayes models. This is likely due to application on SMOTE balancing which has paid off a great deal.

# INSIGHTS
This evaluation compared several models on a dataset containing tweets targeting two brands (Apple and Google) with three sentiment classes: positive, negative, and neutral. The dataset exhibited class imbalance, and most models struggled to differentiate between positive and neutral sentiments due to the similarity in the vocabulary used for these two classes. Above is a summary of the results and an explanation of the metrics used.

# BEST MODEL

The Logistic Regression model with TF-IDF Vectorizer and SMOTE emerged as the best-performing model:

F1 Score: 0.6512, the highest among all models, indicating a better balance between precision and recall.
ROC AUC Score: 0.7519, the highest separability score across classes.
Standard Deviation of ROC AUC: 0.0064, showing relatively consistent performance across folds.
This model benefited from the use of:

TF-IDF Vectorizer: Which prioritized distinguishing terms based on importance.

SMOTE: Which addressed the class imbalance by oversampling the minority class.

# Limitations

Despite its overall performance, the model faced difficulty in distinguishing positive and neutral sentiments, likely due to overlapping vocabulary in tweets related to these sentiments.
Explanation of Metrics

Average F1 Score:

The F1 Score is the harmonic mean of precision (how many predicted positives are correct) and recall (how many actual positives are correctly identified).

It is particularly useful in imbalanced datasets, as it balances the trade-off between false positives and false negatives.

Average ROC AUC Score:

The Area Under the Receiver Operating Characteristic (ROC) Curve measures the model's ability to distinguish between classes.
A higher ROC AUC indicates better separability of classes, irrespective of threshold selection.
Standard Deviation of ROC AUC:
Measures the variability of the ROC AUC score across cross-validation folds.
A lower standard deviation indicates more stable performance.

# Conclusions
The Logistic Regression model with TF-IDF Vectorizer and SMOTE is recommended as the best model for sentiment analysis on this dataset. While it achieved the highest F1 Score and ROC AUC, the presence of class imbalance and the similarity of words in positive and neutral tweets remained challenges for all models.

# Recommendations

Based on the analysis of the sentiment dataset and the performance of multiple models, the following recommendations are provided to optimize sentiment analysis efforts for the business:

Adopt the Best Performing Model The Logistic Regression model with TF-IDF Vectorizer and SMOTE is the most suitable for sentiment analysis due to its strong performance metrics:

F1 Score: 0.6512 (indicating good balance between precision and recall).

ROC AUC: 0.7519 (showing strong ability to distinguish between sentiments).

Consistency: Low standard deviation of 0.0064, ensuring reliable predictions across data samples.

This model is recommended for deployment to analyze customer feedback related to Apple and Google products. Its ability to handle imbalanced data ensures that underrepresented sentiments, such as negative feedback, are not overlooked.

Address Challenges with Positive and Neutral Sentiments The analysis revealed that all models struggled to differentiate between positive and neutral tweets. This is due to the overlap in vocabulary used in these sentiments. To address this:

### Actionable Insights: 

Review common keywords in tweets classified as positive or neutral and refine them. For example, terms like "good" and "okay" may need additional context to ensure correct classification.

### Enhanced Data Collection:

Better classification on sentiment labels can be achieved by collecting more diverse and detailed data. This can help in distinguishing subtle differences in sentiment expressions. For example, collecting tweets with stronger emotional language can improve model performance.

### Different Data Source:

Consider using additional data sources beyond Twitter to capture a wider range of sentiments and expressions. This can provide a more comprehensive view of customer feedback and improve sentiment analysis accuracy.
