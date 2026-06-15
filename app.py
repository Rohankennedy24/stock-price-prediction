from flask import Flask, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load Dataset
df = pd.read_csv("dataset/cleaned_NVDA.csv")

# Load Random Forest Model
rf_model = joblib.load(
    "models/random_forest_model.pkl"
)

@app.route('/')

def home():

    latest = df[
        ['open', 'high', 'low', 'volume']
    ].iloc[-1:]

    # RF Prediction
    rf_prediction = rf_model.predict(
        latest
    )[0]

    # Your LSTM Prediction
    lstm_prediction = 205.02

    # Ensemble
    ensemble_prediction = (
        rf_prediction +
        lstm_prediction
    ) / 2

    current_price = (
        df['close'].iloc[-1]
    )

    # Sentiment Result
    market_sentiment = "POSITIVE"

    # Confidence Score
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

    # Risk Score
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

    # Recommendation

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

    return render_template(
    "index.html",
    current=round(current_price,2),
    rf=round(rf_prediction,2),
    lstm=round(lstm_prediction,2),
    ensemble=round(ensemble_prediction,2),
    sentiment=market_sentiment,
    confidence=round(confidence,2),
    risk=risk,
    recommendation=recommendation,

    chart_labels=[
        "Current",
        "RF",
        "LSTM",
        "Ensemble"
    ],

    chart_values=[
        round(current_price,2),
        round(rf_prediction,2),
        round(lstm_prediction,2),
        round(ensemble_prediction,2)
    ]
)

if __name__ == '__main__':
    app.run(debug=True)