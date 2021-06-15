# Sentiment-Analysis


Sentiment-Analysis

Sentiment Analysis is one of the most used branches of Natural language processing. With the help of Sentiment Analysis, we humans can determine whether the text is showing positive or negative sentiment and this is done using both NLP and machine learning. Sentiment Analysis is also called as Opinion mining. In this article, we will learn about NLP sentiment analysis in python.

From reducing churn to increase sales of the product, creating brand awareness and analyzing the reviews of customers and improving the products, these are some of the vital application of Sentiment analysis. Here, we will implement a machine learning model which will predict the sentiment of customer reviews and we will cover below-listed topics



1. The problem statement:
2. Feature extraction and Text pre-processing
3. Advance text preprocessing
4. Sentiment Analysis by building a machine learning model

Sentiment Analysis using Naive Bayes:
We will be implementing Naive Bayes for sentiment analysis on tweets. Given a tweet, you will decide if it has a positive sentiment or a negative one. Specifically we will cover:

* Training a naive bayes model on a sentiment analysis task
* Test using our model
* Compute ratios of positive words to negative words
* Do some error analysis
* Predict on your own tweet
* We will use the ratio of probabilities between positive and negative sentiments and this approach gives us simpler formulas for these 2-way classification tasks.

Feature extraction and Text pre-processing:
Machines can not understand English or any text data by default. The text data needs a special preparation before you can give text data to the machine to predict something out of it. That special preparation includes several steps such as removing stops words, correcting spelling mistakes, removing meaningless words, removing rare words and many more.

The first step of preparing text data is applying feature extraction and basic text pre-processing. In feature extraction and basic text pre-processing there several steps as follows,

1. Removing Punctuations
2. Removing HTML tags
3. Special Characters removal
4. Removing AlphaNumeric words
5. Tokenization
6. Removal of Stopwords
7. Lower casing
8. Lemmatization

Building NLP sentiment analysis Machine learning model:
Now last the part of the NLP sentiment analysis is to create Machine learning model. In this article, we will use the Naive Bayes classification model. I have written a separate post onNaive Bayes classification model, do read if you not familiar with the topic.

As of now, we have two vectors i.e. X and Y. The first step to create a machine learning model is that splitting the dataset into the Training set and Test set. Using the training set we will create a Naive Bayes classification model. Then With the test set can check the performance of a Naive Bayes classification model.

=> In the below code, first we have imported thetrain_test_split API to split the vectors into test and traing set.

=> We have importedGaussianNB() class to create a Naive Bayes classification model.

=> After creating the Naive Bayes classification model, then we will fit the training set into the Naive Bayes classifier. Open the file nlp.py and write down below code into it.

NLP sentiment analysis In Action:
Now that our model is ready to predict the sentiments based on the Reviews, so why not write a code to test it? By doing this we will understand how well our model is predicting the result and that our end goal as well. So the steps are very straight forward here,

=> First we have createdpredictNewReview() function, which will ask to write a review in CMD and then it will use the above-created classifier to predict the sentiment.

=> As soon aspredictNewReview() function will get a new review it will do all the text cleaning process usingdoTextCleaning() function.

=> Once the text cleaning is performed, then using BOW model transform we will convert the Review the numeric vector.

=> After the conversion, the Naive Bayes classification model can be used to predict the result using classifier.predict() method. Open the file nlp.py and write down below code into it.

Conclusion:
We were able to convert text to numeric vectors and then we used the Machine learning model to predict the sentiment. Here we used the Naive Bayes classification model to predict the sentiment of any given review.

