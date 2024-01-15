# Importing all necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import streamlit as st
import plotly.express as px
import requests

scaler = MinMaxScaler(feature_range=(0, 1))

# HEADER (INPUT)
st.title('Stock Price Prediction')

ticker = st.sidebar.text_input('Ticker', 'AAPL')
start__date = st.sidebar.date_input('Start Date')
end__date = st.sidebar.date_input('End Date')

# Fetching the data from Yahoo Finance
try:
    start_date_str = start__date.strftime('%Y-%m-%d')
    end_date_str = end__date.strftime('%Y-%m-%d')
    df = yf.download(ticker, start=start_date_str, end=end_date_str)
except Exception as e:
    st.write(f"Error fetching data from Yahoo Finance: {str(e)}")
    st.stop()

# Describing data
st.subheader("Statistical Data")
st.write(f'Data from {start_date_str} to {end_date_str}')
st.write(df.describe())

# Dynamic Chart:
try:
    figure = px.line(df, x=df.index, y=df['Close'], title=ticker)
    st.plotly_chart(figure)
except Exception as e:
    st.write(f"Error plotting dynamic chart: {str(e)}")

# Information for the prediction: Pricing table & News column:
pricing_data, news = st.tabs(["Pricing Data","Top 10 news"])

# 1 -> Showing Stock Detailed Price:
with pricing_data:
    try:
        st.write('Price')
        data2 = df
        data2['% Change'] = df['Adj Close'] / df['Adj Close'].shift(1) - 1
        data2.dropna(inplace=True)
        st.write(data2)
        annual_return = data2['% Change'].mean() * 252 * 100
        st.write("Annual Return is : ", annual_return, "%")
        stdev = np.std(data2['% Change']) * np.sqrt(252)
        st.write('Standard Deviation is : ', stdev * 100, '%')
        st.write('Risk Adj. Return is : ', annual_return / (stdev * 100))
    except Exception as e:
        st.write(f"Error displaying pricing data: {str(e)}")

# 2 -> Stock News Fetching From API (NEWS API).
with news:
    try:
        st.subheader("News")
        api_key = 'c4881186704b40b8948feda57a583574'
        input_ticker = ticker  # Replace with the desired stock symbol
        url = f'https://newsapi.org/v2/everything?q={input_ticker}&apiKey={api_key}'
        response = requests.get(url)

        if response.status_code == 200:
            news_data = response.json()
            articles = news_data['articles']
            for i, article in enumerate(articles[:10]):
                st.write(f'News {i + 1}:')
                st.write(f'Title: {article["title"]}')
                st.write(f'Published at: {article["publishedAt"]}')
                st.write(f'Description: {article["description"]}')
                st.write(f'URL: {article["url"]}')
                st.write('\n')
        else:
            st.write(f'Error fetching news. Status code: {response.status_code}')
    except Exception as e:
        st.write(f"Error fetching news data: {str(e)}")

# MODEL PART FOR PREDICTION -> splitting data training and testing in ratio 70:30
try:
    if len(df) < 100:
        st.warning("Insufficient data for prediction. Please choose a longer date range.")
    else:
        data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.7)])
        data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.7):int(len(df))])

        scalar = MinMaxScaler(feature_range=(0, 1))
        data_training_array = scalar.fit_transform(data_training)

        model = load_model('keras_model.madeforPBL')

        # Testing part only for the previous 100 days
        past_100_days = data_training.tail(100).copy()
        final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

        # Now we need to scale down the data using min-max scaler
        input_data = scalar.fit_transform(final_df)

        if len(input_data) < 100:
            st.warning("Insufficient data for prediction. Please choose a longer date range.")
        else:
            # Now we need to create testing arrays
            x_test = []
            y_test = []

            # Using time series
            for i in range(100, input_data.shape[0]):
                x_test.append(input_data[i - 100:i])
                y_test.append(input_data[i, 0])

            # Transforming these arrays into numpy arrays
            x_test, y_test = np.array(x_test), np.array(y_test)

            # Now we will start making predictions
            y_predicted = model.predict(x_test)

            # We need to scale up the value back
            a = scalar.scale_
            scale_factor = 1 / a

            # We will multiply training and testing data by the scale factor
            y_predicted = y_predicted * scale_factor
            y_test = y_test * scale_factor

            # Visualization chart for prediction trends in stock
            st.subheader('Closing Price Vs Time Chart')
            fig = plt.figure(figsize=(12, 6))
            plt.plot(df.Close)
            st.pyplot(fig)

            # 100 days moving average
            st.subheader('Closing Price vs Time In 100 Days')
            ma100 = df.Close.rolling(100).mean()
            fig = plt.figure(figsize=(12, 6))
            plt.plot(ma100, 'r', label="100 Days Avg.")
            plt.plot(df.Close, label='Price')
            plt.legend()
            st.pyplot(fig)

            # 200 days moving average
            st.subheader('Closing Price Vs Time In 100 Days And 200 Days')
            ma100 = df.Close.rolling(100).mean()
            ma200 = df.Close.rolling(200).mean()
            fig = plt.figure(figsize=(12, 6))
            plt.plot(ma100, 'r', label="100 Days Avg.")
            plt.plot(ma200, 'g', label="200 Days Avg.")
            plt.legend()
            plt.plot(df.Close, 'b')
            st.pyplot(fig)

            # Actual vs predicted comparison
            st.subheader('Actual vs Predicted Closing Price')
            fig_actual_vs_predicted = plt.figure(figsize=(12, 6))
            plt.plot(y_test, 'b', label='Actual Price')
            plt.plot(y_predicted, 'r', label='Predicted Price')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()
            st.pyplot(fig_actual_vs_predicted)

except Exception as e:
    st.write(f"Error in prediction model: {str(e)}")
