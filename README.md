# Crypto-Predictor

### Description

I decided to create a cryptocurrency predictor that not only takes in the usual assortment of numerical data, but also receives articles written about the specific currency as input. 

### Capstone Proposal

- Domain Background
  - Finance
- Problem Statement
  - More accurately predict the future price of any stock or cryptocurrency. 
- Datasets and Inputs
  - The articles used will be downloaded from Coinbase's top stories on the specific cryptocurrency's page for each date the numerical data is being used
  - The numerical data will be downloaded from [Coin Metrics](https://coinmetrics.io/data-downloads/).
- Solution Statement
  - The above problem will be solved through the use of articles written about the stock or cryptocurrency in addition to other numerical data as input.
- Benchmark Model
  - The solution will be compared to a model that doesn't take in articles as input.
- Evaluation Metrics
  - The solution can be measured by predicting next day stock price's and comparing to the actual price's.
- Project Design
  - The model used for predicting future stock prices will be AWS's DeepAR Forecasting. PCA may need to be performed here as well, due to the large amount of metrics in the dataset. AWS's BlazingText will be used to perform a form of sentiment analysis when comparing the article to the next several day's stock performance. This will be done seperate from the initial DeepAR model. The sentiment analysis from 1 week to one month's worth of article's will then be added to the rest of the numerical data and a new DeepAR model will be trained and compared to the initial one. 
