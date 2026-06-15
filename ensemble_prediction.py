rf_prediction = 230.8

lstm_prediction = 228.4

ensemble = (
    rf_prediction +
    lstm_prediction
)/2

print(
    "Final Prediction:",
    ensemble
)