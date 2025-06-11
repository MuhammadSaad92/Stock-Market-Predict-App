This is an interactive web application built using Streamlit, TensorFlow/Keras, and yFinance, designed to visualize, analyze,
and predict stock prices of selected companies listed on the Pakistan Stock Exchange (PSX). It leverages historical stock data and 
a trained LSTM deep learning model to provide predictive insights into future stock prices and investment recommendations.

🔧 Features
✅ User Interface
Streamlit-powered Web UI with sidebar inputs
Company selection dropdown for PSX stocks
Custom date range selection with validation
Embedded image/logo at the top
Center-aligned header with stock market theme

📊 Data Visualization
Fetches real-time historical data using yfinance
Displays raw stock data table for the selected range
Line chart of closing prices
Visualization of predicted vs actual prices
Colorful area charts using matplotlib.fill_between for clarity
Metrics such as:
Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)

🔍 Model Details
Uses a pre-trained LSTM model (saad_keras_model.h5)
Normalizes input data using MinMaxScaler
Performs 80/20 train-test split for evaluation
Predicts closing stock prices and compares with actuals

📅 Future Prediction
Simulates the next 365 days of future stock prices
Adds realistic randomness to simulate market volatility
Generates:
Future price line chart
Future price prediction table

💡 Investment Suggestion Logic
Calculates 200-day SMA (Simple Moving Average)
Evaluates:
Recent short-term price movement
Long-term SMA trend

Displays intuitive guidance:
✅ Invest
⚠️ Consider Investing
❌ Do Not Invest

Offers clear, user-friendly reasons for each suggestion
