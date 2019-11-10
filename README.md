# Crypto-Predictor

### Description

I decided to create a cryptocurrency predictor that not only takes in the usual assortment of numerical data, but also receives articles written about the specific currency as input. 

### Capstone Proposal

- Domain Background
  - Finance
- Problem Statement
  - More accurately predict the future price of any stock or cryptocurrency through a classification model that takes in a variety of features, including articles, as input and produces the predicted closing stock price in USD for the following day.
  - Past work I'll be referencing in this project comes from the most popular cryptocurrency predictor found on Github, khuangaf's [CryptocurrencyPrediction](https://github.com/khuangaf/CryptocurrencyPrediction). A key difference right away between the data used to train khuangaf's model's and the data used to train my model's are the number of features, with khuangaf's data coming from [poloniex](https://docs.poloniex.com/#introduction) and containing only 5 features. Please see khuangaf's [Data Collection Notebook](https://github.com/khuangaf/CryptocurrencyPrediction/blob/master/DataCollection.ipynb). Whereas the data my model's will be trained with contains 32 features, 33 when including the articles sentiment analysis. Please see a detailed explanation of all the features [here](https://coinmetrics.io/community-data-dictionary/).
  - The data used by khuangaf consists of 256 steps every 5 minutes in the past as input and 16 steps per every five minutes in the future as output. The data I gathered from Coin Metrics consist of 90 steps, 1 per day, in the past, and an output of 1 day with XGBoost and 5 days with DeepAR.
  - The models khuangaf constructed were custom CNN, GRU, and LSTM models constructed using keras. I will be training models using AWS's XGBoost, DeepAR, Blazing Text, and PCA built in algorithms.
  - Past work to be cited for the sentiment analysis feature used in this project is [Jason Yip's aritcle](https://towardsdatascience.com/https-towardsdatascience-com-algorithmic-trading-using-sentiment-analysis-on-news-articles-83db77966704). In this project Jason used Natural Language Toolkit's (nltk), SentimentIntensityAnalyzer to obtain a sentiment score on articles written about Facebook. If the score was above 0.5, Jason simulated purchasing 10 shares of Facebook stock, if it was below 0.5 he sold 10 shares. Jason ended with $99,742 after starting with $100,000. The articles Jason used were gathered from [Business Time](https://www.businesstimes.com.sg/search/facebook?page=1).
  - The sentiment analysis model I train will be trained on cryptocurrency articles, unlike the nltk's SentimentIntensityAnalyzer which was trained for social media posts. 
- Datasets and Inputs
  - The articles used will be downloaded from Coinbase's top stories on the specific cryptocurrency's page for each date the numerical data is being used.
  - The numerical data will be downloaded from [Coin Metrics](https://coinmetrics.io/data-downloads/). A lot of the features in the dataset are means and medians of the same feature, making PCA for dimension reduction promising.
- Solution Statement
  - The above problem will be solved through the use of articles written about the stock or cryptocurrency in addition to other numerical data as input.
- Benchmark Model
  - Khuangaf's model will be considered the benchmark, although the key differences mentioned above make it unable to be directly compared. Instead the following models will be compared:
    - XGBoost Model trained with all features from Coin Metrics
    - DeepAR Model trained with all features from Coin Metrics
    - XGBoost Model trained using PCA to reduce dimensionality
    - DeepAR Model trained using PCA to reduce dimensionality
    - XGBoost Model trained with PCA and sentiment analysis
    - DeepAR Model trained with PCA and sentiment analysis
- Evaluation Metrics
  - The purpose of the models I will be creating aim to explore the profitability of short term investments in cryptocurrency using classification models to predict next day closing prices. The 22% tax rate will be applied to all profits, and Coinbase's 3.0% (1.5% at time of purchase & 1.5% at time of sale) fee will be applied to total investments.
  - I will evaluate how successful the model is by simulating the trading of an initial investment of $1000 through one month. I will simulate the following thresholds for purchasing and selling stock:
    - Purchase at any predicted growth; Sell at any predicted loss
    - Purchase at 3% predicted growth; Sell at any predicted loss
    - Purchase at 3% predicted growth; Sell at 1.5% predicted loss
    - Purchase at 5% predicted growth; Sell at any predicted loss
    - Purchase at 5% predicted growth; Sell at 1.5% predicted loss
  - A model will be deemed successful if it is able to average 10% profit per month, over 3 months with one of the strategies above.
- Project Design
  - The model used for predicting next 5 day stock prices will be AWS's DeepAR Forecasting. The model used for predicting next day stock prices will be XGBoost. AWS's BlazingText will be used to perform a form of sentiment analysis when comparing the article to the next several day's stock performance. This will be done seperate from the initial DeepAR model. The sentiment analysis from 1 week to one month's worth of article's will then be added to the rest of the numerical data and a new DeepAR model will be trained and compared to the initial one. 
