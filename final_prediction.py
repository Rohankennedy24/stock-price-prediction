import pandas as pd
import joblib
import numpy as np

# --------------------------------
# LOAD DATA
# --------------------------------

df = pd.read_csv(
    "dataset/cleaned_NVDA.csv"
)

# --------------------------------
# RANDOM FOREST
# --------------------------------

rf_model = joblib.load(
    "models/random_forest_model.pkl"
)

latest = df[
    ['open','high','low','volume']
].iloc[-1:]

rf_prediction = rf_model.predict(
    latest
)[0]

# --------------------------------
# LSTM PREDICTION
# --------------------------------

# Replace with your LSTM result

lstm_prediction = 205.02

# --------------------------------
# ENSEMBLE
# --------------------------------

ensemble_prediction = (
    rf_prediction +
    lstm_prediction
) / 2

# --------------------------------
# CURRENT PRICE
# --------------------------------

current_price = (
    df['close']
    .iloc[-1]
)

# --------------------------------
# MARKET SENTIMENT
# --------------------------------

market_sentiment = "POSITIVE"

# --------------------------------
# CONFIDENCE SCORE
# --------------------------------

difference = abs(
    rf_prediction -
    lstm_prediction
)

confidence = max(
    50,
    100 -
    (
        difference/current_price
    ) * 100
)

# --------------------------------
# RISK SCORE
# --------------------------------

volatility = (
    df['close']
    .pct_change()
    .std()
) * 100

if volatility < 2:
    risk = "LOW"

elif volatility < 5:
    risk = "MEDIUM"

else:
    risk = "HIGH"

# --------------------------------
# RECOMMENDATION
# --------------------------------

price_change = (
    (
        ensemble_prediction -
        current_price
    )
    / current_price
) * 100

if (
    market_sentiment == "POSITIVE"
    and price_change > 2
):
    recommendation = "BUY"

elif (
    market_sentiment == "NEGATIVE"
):
    recommendation = "SELL"

else:
    recommendation = "HOLD"

# --------------------------------
# OUTPUT
# --------------------------------

print("\n========== AI STOCK ADVISOR ==========")

print(
    "Current Price :",
    round(current_price,2)
)

print(
    "RF Prediction :",
    round(rf_prediction,2)
)

print(
    "LSTM Prediction :",
    round(lstm_prediction,2)
)

print(
    "Ensemble Prediction :",
    round(ensemble_prediction,2)
)

print(
    "Market Sentiment :",
    market_sentiment
)

print(
    "Confidence Score :",
    round(confidence,2),
    "%"
)

print(
    "Risk Level :",
    risk
)

print(
    "Recommendation :",
    recommendation
)

print("====================================")