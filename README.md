# Crypto-Predictor

### Description

I decided to create a cryptocurrency predictor that takes the term day trading literally.  

### Capstone Proposal

- Domain Background
  - Finance
    - Day trading is a very risky and time consuming practice. Day traders typically make multiple trades a day.  
    - This project aims to create a model that accurately predicts the end of day price, each day.
    - The most prominent use case of this project would be to make money through only end of day trades. This would make "day trading" an attainable option for somebody with a full time job. 
- Problem Statement
  - More accurately predict the future price of any stock or cryptocurrency through a classification model that takes in a variety of features as input and produces the predicted closing stock price in USD for the following day.
  - Past work I'll be referencing in this project comes from the most popular cryptocurrency predictor found on Github, khuangaf's [CryptocurrencyPrediction](https://github.com/khuangaf/CryptocurrencyPrediction). A key difference right away between the data used to train khuangaf's model's and the data used to train my model's are the number of features, with khuangaf's data coming from [poloniex](https://docs.poloniex.com/#introduction) and containing only 5 features. Please see khuangaf's [Data Collection Notebook](https://github.com/khuangaf/CryptocurrencyPrediction/blob/master/DataCollection.ipynb). Whereas the data my model's will be trained with contains 32 features. Please see a detailed explanation of all the features [here](https://coinmetrics.io/community-data-dictionary/). The data my model uses is able to have much more features because the data my model is using is gathered at the end of the day.
  - The data used by khuangaf consists of 256 steps every 5 minutes in the past as input and 16 steps per every five minutes in the future as output. The data I gathered from Coin Metrics consist of 1095 steps (3 years), 1 per day, in the past, and an output of 1 day with XGBoost and 30 days with DeepAR.
  - The models khuangaf constructed were custom CNN, GRU, and LSTM models constructed using keras. I will be training models using AWS's XGBoost, DeepAR, and PCA built in algorithms.
- Datasets and Inputs
  - The data will be downloaded from [Coin Metrics](https://coinmetrics.io/data-downloads/). A lot of the features in the dataset are means and medians of the same feature, making PCA for dimension reduction promising. Again because this data is obtained at the end of day, it is able to have way more features than the data gathered for khuangaf's model.
- Solution Statement
  - The above problem will be solved by refining the models and data until it becomes profitable. 
  - Hyperparameter tuning will be performed on the xgboost models.  
  - The features dimensionality will not be reduced on any models initially. PCA analysis will be tested on the xgboost models. If PCA analysis is able lower the test score, then it will also be tested on the DeepAR model. Dimensionality will also be reduced through deductive reasoning.   
- Benchmark Model
  - Khuangaf's model will be considered the benchmark, although the key differences mentioned above make it unable to be directly compared. Instead the following models will all be compared with eachother:
    - XGBoost Model trained with all features from Coin Metrics
    - DeepAR Model trained with all features from Coin Metrics
    - XGBoost Model trained with PCA.
    - If dimensionality reduction through PCA generates a lower test loss for XGBoost, then it will also be done for a DeepAR model.
    - XGBoost Model trained with dimensionality reduction through deductive reasoning.
    - DeepAR Model trained with dimensionality reduction through deductive reasoning.
- Evaluation Metrics
  - The purpose of the models I will be creating aim to explore the profitability of short term investments in cryptocurrency using classification models to predict next day closing prices. 
  - I will evaluate how successful the model is by simulating the trading of an initial investment of $20000 through the test data set.
  - A model will be deemed successful if it has a test loss of less than one USD and is able to average 10% profit per month tested. 
  - The test loss will be simply calculated by taking the mean of the absolute value of the difference between the target price and the predicted price.
- Project Design
  - The model used for predicting next 30 day stock prices will use AWS's built-in algorithm DeepAR Forecasting. The model used for predicting next day stock prices will use AWS's XGBoost. 

  - Machine Learning Workflow

    - This project predicts next day and next 30 days closing price of Litecoin through the following steps:
      * Loading and exploring the data
      * Data cleaning and pre-processing
      * Dimensionality reduction of coinmetric's dataset's 32 features with PCA
      * Feature engineering with data transformation
      * Dimensionality reduction through deductive reasoning
      * Uploading data to S3
      * Instantiating and training a DeepAR estimator as well as XGBoost estimator
      * Deploying a model and creating a predictor
      * Evaluating the predictor through simulating buying and selling Litecoin based on model predictions over 30 day's
      * Evaluating the predictor by calculating the test loss
