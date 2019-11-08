# Crypto-Predictor

### Description

I decided to create a cryptocurrency predictor that not only takes in the usual assortment of numerical data, but also receives articles written about the specific currency as input. 

### Capstone Proposal

- Domain Background
  - Finance
- Problem Statement
  - More accurately predict the future price of any stock or cryptocurrency through a classification model that takes in a variety of features, including articles, as input and produces the predicted closing stock price in USD for the following day.
  - Past work I'll be referencing in this project comes from the most popular cryptocurrency predictor found on Github, khuangaf's [CryptocurrencyPrediction](https://github.com/khuangaf/CryptocurrencyPrediction). A key difference right away between the data used to train khuangaf's model's and the data used to train my model's are the number of features, with khuangaf's data coming from [poloniex](https://docs.poloniex.com/#introduction) and containing only 5 features. Please see khuangaf's [Data Collection Notebook](https://github.com/khuangaf/CryptocurrencyPrediction/blob/master/DataCollection.ipynb). Whereas the data my model's will be trained with contains 32 features, 33 when including the articles sentiment analysis. Please see a detailed explanation of all the features [here](https://coinmetrics.io/community-data-dictionary/).
  - The data used by khuangaf consists of 256 steps every 5 minutes in the past as input and 16 steps per every five minutes in the future as output. The data I gathered from Coin Metrics consist of 90 steps, 1 per day, in the past, and an output of 1 day in the future. I will also train models that go 60 and 30 days back as well. 
  - The models khuangaf constructed were custom CNN, GRU, and LSTM models constructed using keras. I will be training AWS's built in models.
  - The purpose of the models I will be creating aim to explore the profitability of short term investments in cryptocurrency using classification models to predict next day closing prices. The 22% tax rate will be applied to all profits, and Coinbase's 1.5% fee will be applied to total investments.
- Datasets and Inputs
  - The articles used will be downloaded from Coinbase's top stories on the specific cryptocurrency's page for each date the numerical data is being used.
  - The numerical data will be downloaded from [Coin Metrics](https://coinmetrics.io/data-downloads/).
- Solution Statement
  - The above problem will be solved through the use of articles written about the stock or cryptocurrency in addition to other numerical data as input.
- Benchmark Model
  - The solution will be compared to a model that doesn't take in articles as input.
- Evaluation Metrics
  - The solution can be measured by predicting next day stock price's and comparing to the actual price's.
- Project Design
  - The model used for predicting future stock prices will be AWS's DeepAR Forecasting. PCA may need to be performed here as well, due to the large amount of features in the dataset. AWS's BlazingText will be used to perform a form of sentiment analysis when comparing the article to the next several day's stock performance. This will be done seperate from the initial DeepAR model. The sentiment analysis from 1 week to one month's worth of article's will then be added to the rest of the numerical data and a new DeepAR model will be trained and compared to the initial one. 
