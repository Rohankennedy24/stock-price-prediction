import joblib
import pandas as pd

model = joblib.load(
    "models/random_forest_model.pkl"
)

df = pd.read_csv(
    "dataset/cleaned_NVDA.csv"
)

latest = df[['open','high','low','volume']].iloc[-1:]

prediction = model.predict(latest)

print(
    "Predicted Next Closing Price:",
    prediction[0]
)