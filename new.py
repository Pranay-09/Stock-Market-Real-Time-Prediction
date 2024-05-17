import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model
from TSInterpret.InterpretabilityModels.Saliency.TSR import TSR
from datetime import date, timedelta
import pandas_ta as ta
from sklearn.metrics import mean_squared_error, r2_score
from keras import backend as K
def mse_scorer(model, X, y):
    y_pred = model.predict(X)
    return -mean_squared_error(y, y_pred)

import plotly.graph_objects as go

# Add candlestick plot
def create_candlestick_plot(data):
    candlestick = go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Candlestick'
    )

    layout = go.Layout(
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        title=f'{selected_ticker} Candlestick Chart',
        xaxis_rangeslider_visible=False
    )

    fig = go.Figure(data=[candlestick], layout=layout)
    st.plotly_chart(fig)

# Add volume plot
def create_volume_plot(data):
    volume = go.Bar(
        x=data.index,
        y=data['Volume'],
        name='Volume'
    )

    layout = go.Layout(
        xaxis_title='Date',
        yaxis_title='Volume',
        title=f'{selected_ticker} Trading Volume',
        xaxis_rangeslider_visible=False
    )

    fig = go.Figure(data=[volume], layout=layout)
    st.plotly_chart(fig)


def predict_stock_price(stock, start_date, end_date, future_days):
    start = start_date
    end = end_date
    data = yf.download(stock, start, end)
    data.reset_index(inplace=True)

    ma_100_days = data.Close.rolling(100).mean()
    ma_200_days = data.Close.rolling(200).mean()

    data.dropna(inplace=True)

    data_train = pd.DataFrame(data.Close[0: int(len(data) * 0.80)])
    data_test = pd.DataFrame(data.Close[int(len(data) * 0.80): len(data)])

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_train_scale = scaler.fit_transform(data_train)

    x = []
    y = []

    for i in range(100, data_train_scale.shape[0]):
        x.append(data_train_scale[i - 100:i])
        y.append(data_train_scale[i, 0])

    x, y = np.array(x), np.array(y)
    print(x.shape)

    # Check if the model file exists
    model_file = 'Stock_Market_Predictor.keras'
    # if os.path.isfile(model_file):
    #     # Load the saved model
    #     model = load_model(model_file)
    # else:
        # Train the model
    print("Training model")
    model = Sequential()
    model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=((x.shape[1], 1))))
    model.add(Dropout(0.2))
    model.add(LSTM(units=60, activation='relu', return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(units=80, activation='relu', return_sequences=True))
    model.add(Dropout(0.4))
    model.add(LSTM(units=120, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x, y, epochs=1, batch_size=32, verbose=1)

    # Save the trained model
    model.save(model_file)

    pas_100_days = data_train.tail(100)
    data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
    data_test_scale = scaler.fit_transform(data_test)

    # Calculate technical indicators
    data['RSI'] = ta.rsi(data['Close'], length=14)
    data['SMA_20'] = data['Close'].rolling(20).mean()
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()



    x = []
    y = []

    for i in range(100, data_test_scale.shape[0]):
        x.append(data_test_scale[i - 100:i])
        y.append(data_test_scale[i, 0])
    x, y = np.array(x), np.array(y)

    y_predict = model.predict(x)

    scale = 1 / scaler.scale_
    y_predict = y_predict * scale
    y = y * scale

    # Predict future stock prices
    last_100_days = data.tail(100)
    last_100_days_scaled = scaler.transform(last_100_days['Close'].values.reshape(-1, 1))
    future_data = []

    for i in range(future_days):
        x_future = last_100_days_scaled[i:100 + i, 0]
        if len(x_future) < 100:
            x_future = np.pad(x_future, (0, 100 - len(x_future)), 'edge')
        x_future = x_future.reshape(1, 100, 1)
        future_price = model.predict(x_future)
        future_price = scaler.inverse_transform(future_price)
        future_data.append(future_price[0, 0])


    st.subheader(f'Stock Price Prediction for {stock}')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(y_predict, 'r', label='Predicted Price')
    ax.plot(y, 'g', label='Original Price')
    ax.plot(future_data, 'b', label='Future Prediction')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)

    # Calculate model performance metrics
    r2 = r2_score(y, y_predict)
    mse = mean_squared_error(y, y_predict)
    st.write(f"R-squared: {r2:.4f}")
    st.write(f"Mean Squared Error: {mse:.4f}")

    st.subheader('Technical Indicators')

    fig1, ax1 = plt.subplots()
    ax1.plot(data['RSI'])
    ax1.set_xlabel('Date')
    ax1.set_ylabel('RSI')
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.plot(data['SMA_20'], label='SMA_20')
    ax2.plot(data['EMA_50'], label='EMA_50')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Value')
    ax2.legend()
    st.pyplot(fig2)

    create_candlestick_plot(data)

    # Create volume plot
    create_volume_plot(data)

    int_mod=TSR(model, x.shape[-2],x.shape[-1], method='IG',mode='time')
    item= np.array([data_test.iloc[0:100]])
    label=int(np.argmax(y[0]))

    exp=int_mod.explain(item,labels=label,TSR =True)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(int_mod.plot(np.array([data_test.iloc[0:100]]),exp))


st.title('Stock Market Predictor')

# Get the list of stock tickers
tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()

selected_ticker = st.selectbox('Select a stock ticker', tickers)

today = date.today().strftime("%Y-%m-%d")
start_date = st.date_input("Start Date", value=pd.to_datetime('2012-01-01'), max_value=pd.to_datetime(today))
end_date = st.date_input("End Date", value=pd.to_datetime(today))

future_days = st.number_input("Number of future days to predict", min_value=1, max_value=365, value=30)

print(start_date, end_date, future_days,selected_ticker)

if st.button('Predict Stock Price'):
    predict_stock_price(selected_ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), future_days)

