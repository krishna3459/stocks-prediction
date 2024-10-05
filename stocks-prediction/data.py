import argparse
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

message = "The Stock-Predictor-V4 Project has reached its full potential and will no longer receive updates.\nTo add new functionalities or prediction logic,\nwe welcome third-party pull requests from developers with the necessary expertise.\nThank you for your support and understanding."

def cpugpu():
    import tensorflow as tf
    # Check if GPU is available and print the list of GPUs
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            print(f"Found GPU: {gpu}")
    else:
        print("No GPU devices found.")
    
    if gpus:
        try:
            # Enable memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            # Specify the GPU device to use (e.g., use the first GPU)
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            
            # Test TensorFlow with a simple computation on the GPU
            with tf.device('/GPU:0'):
                a = tf.constant([1.0, 2.0, 3.0])
                b = tf.constant([4.0, 5.0, 6.0])
                c = a * b

            print("GPU is available and TensorFlow is using it.")
            print("Result of the computation on GPU:", c.numpy())
        except RuntimeError as e:
            print("Error while setting up GPU:", e)
    else:
        print("No GPU devices found, TensorFlow will use CPU.")

def install_dependencies():
    import os
    import platform
    import subprocess
    import time
    import sys

    print("\n--------------------------------------")
    system = platform.system()    
    print("OS: ", system)
    print("--------------------------------------\n")

    def install_dependencies():
        print("Installing Python dependencies...")
        start_time = time.time()
        packages = [
            "pandas",
            "ta",
            "yfinance",
            "numpy",
            "scikit-learn",
            "matplotlib",
            "tensorflow",
            "statsmodels"
        ]
        total_packages = len(packages)
        progress = 0
        for package in packages:
            progress += 1
            print(f"Installing {package}... ({progress}/{total_packages})")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", package], check=True
            )
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(
            f"Python dependencies installation complete (Time: {elapsed_time:.2f} seconds)"
        )

    if __name__ == "__main__":
        print("Welcome to the SPV4 installation!")
        print("This script will install all the necessary dependencies.\n")
        print("Prior to proceeding, ensure that you have the necessary programs installed to enable TensorFlow to utilize your GPU or GPUs. If you haven't done so yet, you may press CTRL + C now to halt the process.")

        time.sleep(5)

        print("\nPython dependencies installation will now begin.")

        install_dependencies()

        print("Creating 'data' directory...")
        os.makedirs("data", exist_ok=True)

        print("'data' directory created successfully!\n")
        print("SPV4 installation completed successfully!")

        print(message)

def prepare_data():
    import os
    import pandas as pd
    import numpy as np
    import ta
    import matplotlib.pyplot as plt
    import yfinance as yf
    from datetime import datetime
    from statsmodels.tsa.seasonal import seasonal_decompose

    def download_and_prepare_data(ticker_symbol):
        # Function to download and prepare data
        print(f"Downloading data for {ticker_symbol} from Yahoo Finance...")
        
        # Download data using yfinance
        data = yf.download(ticker_symbol, period="max", interval="1d")
        
        df = pd.DataFrame(data)
        
        # Save the DataFrame to a CSV file
        data_file = f"./data/{ticker_symbol}.csv"
        df.to_csv(data_file)
        
        print("Data downloaded and saved to", data_file)

    def preprocess_data(ticker_symbol):
        # Function to preprocess and analyze the data
        print("Preprocessing and analyzing the CSV data...")
        
        # Read the CSV file
        data_file = f"./data/{ticker_symbol}.csv"
        df = pd.read_csv(data_file)
        
        # Calculate technical indicators using ta library
        df["SMA"] = ta.trend.sma_indicator(df["Close"], window=14)
        df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
        df["MACD"] = ta.trend.macd_diff(df["Close"], window_slow=26, window_fast=12, window_sign=9)
        df_bollinger = ta.volatility.BollingerBands(df["Close"], window=20)
        df["upper_band"] = df_bollinger.bollinger_hband()
        df["middle_band"] = df_bollinger.bollinger_mavg()
        df["lower_band"] = df_bollinger.bollinger_lband()
        df["aroon_up"] = ta.trend.aroon_up(df["Close"], window=25)
        df["aroon_down"] = ta.trend.aroon_down(df["Close"], window=25)
        
        open_prices = df["Open"]
        close_prices = df["Close"]

        # Calculate the "Kicking" pattern feature using NumPy
        kicking_pattern = np.zeros_like(open_prices)

        # Loop through the data and check for "Kicking" pattern
        for i in range(1, len(open_prices)):
            if open_prices[i] < open_prices[i-1] and \
            open_prices[i] > close_prices[i-1] and \
            close_prices[i] > open_prices[i-1] and \
            close_prices[i] < close_prices[i-1] and \
            open_prices[i] - close_prices[i] > open_prices[i-1] - close_prices[i-1]:
                kicking_pattern[i] = 100  # Some positive value to indicate the pattern

        # Assign the kicking_pattern as a new column to the DataFrame
        df["kicking"] = kicking_pattern

        # Calculate ATR and SuperTrend
        def calculate_atr(high, low, close, window=14):
            true_ranges = np.maximum.reduce([high - low, np.abs(high - close.shift()), np.abs(low - close.shift())])
            atr = np.zeros_like(high)
            atr[window - 1] = np.mean(true_ranges[:window])
            for i in range(window, len(high)):
                atr[i] = (atr[i - 1] * (window - 1) + true_ranges[i]) / window
            return atr

        df["ATR"] = calculate_atr(df["High"], df["Low"], df["Close"], window=14)

        # Calculate Supertrend signals with reduced sensitivity and using all indicators
        df["upper_band_supertrend"] = df["High"] - (df["ATR"])
        df["lower_band_supertrend"] = df["Low"] + (df["ATR"])

        # Define conditions for uptrend and downtrend based on sensitivity to indicators
        uptrend_conditions = [
            (df["Close"] > df["lower_band_supertrend"]),
            (df["Close"] > df["SMA"]),
            (df["Close"] > df["middle_band"]),
            (df["Close"] > df["MACD"]),
            (df["RSI"] > 50),
            (df["aroon_up"] > df["aroon_down"]),
            (df["kicking"] == 1),  # Assuming "kicking" is an indicator where 1 indicates an uptrend.
            (df["Close"] > df["upper_band_supertrend"])
        ]

        downtrend_conditions = [
            (df["Close"] < df["upper_band_supertrend"]),
            (df["Close"] < df["SMA"]),
            (df["Close"] < df["middle_band"]),
            (df["Close"] < df["MACD"]),
            (df["RSI"] < 50),
            (df["aroon_up"] < df["aroon_down"]),
            (df["kicking"] == -1),  # Assuming "kicking" is an indicator where -1 indicates a downtrend.
            (df["Close"] < df["lower_band_supertrend"])
        ]

        # Set initial signal values to 0
        df["supertrend_signal"] = 0

        # Update signals based on sensitivity to indicators
        df.loc[np.any(uptrend_conditions, axis=0), "supertrend_signal"] = 1
        df.loc[np.any(downtrend_conditions, axis=0), "supertrend_signal"] = -1

        # Decompose the time series data
        result = seasonal_decompose(df["Close"], model="additive", period=365)

        # Add decomposed components to the DataFrame
        df["trend"] = result.trend
        df["seasonal"] = result.seasonal
        df["residual"] = result.resid

        # Concatenate the columns in the order you want
        df2 = pd.concat(
            [
                df["Date"],
                df["Close"],
                df["Open"],
                df["Adj Close"],
                df["Volume"],
                df["High"],
                df["Low"],
                df["SMA"],
                df["MACD"],
                df["upper_band"],
                df["middle_band"],
                df["lower_band"],
                df["supertrend_signal"],
                df["RSI"],
                df["aroon_up"],
                df["aroon_down"],
                df["kicking"],
                df["upper_band_supertrend"],
                df["lower_band_supertrend"],
                df["trend"],
                df["seasonal"],
                df["residual"]
            ],
            axis=1,
        )

                # Fill missing values with 0
        df2.fillna(0, inplace=True)

        # Save the DataFrame to a new CSV file with indicators and decomposed components
        df2.to_csv("data.csv", index=False)

        # Remove consecutive signals in the same direction (less sensitive)
        signal_changes = df["supertrend_signal"].diff().fillna(0)
        consecutive_mask = (signal_changes == 0) & (signal_changes.shift(-1) == 0)
        df.loc[consecutive_mask, "supertrend_signal"] = 0

        # Plot the data
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(12, 8), sharex=True)

        ax1.plot(df["Open"], label="Open")
        ax1.plot(df["Close"], label="Close")
        ax1.plot(df["trend"], label="Trend")
        
        ax1.plot(df["SMA"], label="SMA")
        ax1.fill_between(
            df.index, df["upper_band"], df["lower_band"], alpha=0.2, color="gray"
        )
        ax1.plot(df["upper_band"], linestyle="dashed", color="gray")
        ax1.plot(df["middle_band"], linestyle="dashed", color="gray")
        ax1.plot(df["lower_band"], linestyle="dashed", color="gray")
        ax1.scatter(
            df.index[df["supertrend_signal"] == 1],
            df["Close"][df["supertrend_signal"] == 1],
            marker="^",
            color="green",
            s=100,
        )
        ax1.scatter(
            df.index[df["supertrend_signal"] == -1],
            df["Close"][df["supertrend_signal"] == -1],
            marker="v",
            color="red",
            s=100,
        )
        ax1.legend()

        ax2.plot(df["aroon_up"], label="Aroon Up")
        ax2.plot(df["aroon_down"], label="Aroon Down")
        ax2.legend()

        ax3.plot(df["RSI"], label="RSI")
        ax3.legend()

        ax4.plot(df["seasonal"], label="Seasonal")
        ax4.legend()

        ax5.plot(df["residual"], label="Residual")
        ax5.legend()

        plt.xlim(df.index[0], df.index[-1])

        plt.show()

    if __name__ == "__main__":
        ticker_symbol = input("Enter the ticker symbol (e.g., AAPL for Apple Inc.): ")
        tic = yf.Ticker(ticker_symbol)
        info = tic.get_info()

        print(f"Information of {ticker_symbol}:\n")

        print(info["shortName"],"\n")

        indicator = "UNKNOWN"

        if "recommendationKey" in info:
            if info["recommendationKey"] == "buy":
                indicator = "Buy"
            elif info["recommendationKey"] == "sell":
                indicator = "Sell"
            elif info["recommendationKey"] == "strong buy":
                indicator = "Strong Buy"
            elif info["recommendationKey"] == "hold":
                indicator = "Hold"
            elif info["recommendationKey"] == "underperform":
                indicator = "Underperform"

        print(f"Recommendation Trend is on {indicator}")

        download = input("Do you want to download this Ticker?: ").lower()

        if download == "yes":
            download_and_prepare_data(ticker_symbol)
            preprocess_data(ticker_symbol)
        else:
            print("Exiting Script..")

        print(message)

def train_model():
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    import pandas as pd
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import BatchNormalization, PReLU, LSTM, Dense
    from sklearn.metrics import r2_score, mean_absolute_percentage_error
    from sklearn.preprocessing import MaxAbsScaler

    print("TensorFlow version:", tf.__version__)

    # Define a function to load data (replace 'data.csv' with your data file)
    def load_data( AMZN: str) -> pd.DataFrame:
        data = pd.read_csv(AMZN.csv )
        return data[['Close']]  # Only load the 'Close' column

    # Define reward function
    def get_reward(y_true, y_pred):
        mape = mean_absolute_percentage_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        reward = (((1 - mape) * 0.1) + ((r2) * 1.9)) / 2
        return reward

    # Define a function to create the LSTM model
    def create_LSTM_model() -> Sequential:
        model = Sequential()
        
        model.add(LSTM(units=150, return_sequences=True))
        model.add(PReLU())
        model.add(PReLU())
        model.add(PReLU())
        model.add(LSTM(units=150))
        model.add(PReLU())
        model.add(PReLU())
        model.add(PReLU())

        # Add the final output layer
        model.add(Dense(units=1, activation='linear'))

        return model

    # Load data
    data = load_data("data.csv")

    # Split data into train and test sets
    train_data = data.iloc[:int(0.8*len(data))]
    test_data = data.iloc[int(0.8*len(data)):]

    # Normalize data (only use 'Close' column)
    scaler = MaxAbsScaler()
    train_data_norm = scaler.fit_transform(train_data)
    test_data_norm = scaler.transform(test_data)

    # Define time steps
    timesteps = 100

    # Create sequences of timesteps (only using 'Close' values)
    def create_sequences(data, timesteps):
        X = []
        y = []
        for i in range(timesteps, len(data)):
            X.append(data[i-timesteps:i])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_data_norm, timesteps)
    X_test, y_test = create_sequences(test_data_norm, timesteps)

    # Build and compile the LSTM model
    model = create_LSTM_model()

    # Compile model
    model.compile(optimizer='adam', loss="huber")

    # Training
    epochs = 25
    batch_size = 32

    for i in range(epochs):
        print(f"Epoch {i+1} / {epochs}")
        history = model.fit(X_train, y_train, batch_size=batch_size, validation_data=(X_test, y_test), epochs=1)
        
        # Evaluate the model on the test set
        y_pred_test = model.predict(X_test)
        test_reward = get_reward(y_test, y_pred_test)
        
        print("Test reward:", test_reward)
        
        if i == 0:
            best_reward = test_reward
        
        if test_reward >= best_reward:
            best_reward = test_reward
            print("Model saved!")
            model.save("model.keras")

    # Load the best model and evaluate
    model = tf.keras.models.load_model("model.keras")
    y_pred_test = model.predict(X_test)
    test_reward = get_reward(y_test, y_pred_test)
    test_loss = model.evaluate(X_test, y_test)

    print("Final test reward:", test_reward)
    print("Final test loss:", test_loss)

    print(message)

def eval():
    import os
    import sys
    import pandas as pd
    import numpy as np
    import tensorflow as tf
    from sklearn.preprocessing import MaxAbsScaler
    from tensorflow.keras.models import load_model
    from sklearn.metrics import (
        mean_squared_error,
        r2_score,
        mean_absolute_percentage_error,
    )
    import matplotlib.pyplot as plt

    print("TensorFlow version:", tf.__version__)

    cpugpu()

    # Define reward function
    def get_reward(y_true, y_pred):
        mape = mean_absolute_percentage_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        reward = (((1 - mape)*0.1) + ((r2)*1.9)) / 2
        return reward

    # Load data
    data = pd.read_csv("data.csv")

    # Split data into train and test sets
    test_data = data.iloc[int(0.8 * len(data)) :]

    # Normalize data
    scaler = MaxAbsScaler()
    test_data_norm = scaler.fit_transform(
        test_data[
            [
"Close"
            ]
        ]
    )

    # Define time steps
    timesteps = 100

    # Create sequences of timesteps
    def create_sequences(data, timesteps):
        X = []
        y = []
        for i in range(timesteps, len(data)):
            X.append(data[i - timesteps : i])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    X_test, y_test = create_sequences(test_data_norm, timesteps)

    # Load pre-trained model
    model = load_model("model.keras")
    print("\nEvaluating Model")

    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Calculate the reward
    reward = get_reward(y_test, y_pred)

    # Print evaluation metrics
    print("MAPE:", mape)
    print("MSE:", mse)
    print("R2:", r2)
    print("Reward:", reward)

    # Plot predictions vs. actual values if needed
    plt.plot(y_test, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.legend()
    plt.show()

    print(message)


def fine_tune_model():
    print("Finetuning the model...")
    import os
    import signal

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    