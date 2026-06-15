import pandas as pd
import numpy as np

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

model = load_model(
    "models/lstm_model.h5"
)

df = pd.read_csv(
    "dataset/cleaned_NVDA.csv"
)

data = df[['close']].values
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)
last_60 = scaled[-60:]

X = np.array([last_60])
prediction = model.predict(X)
prediction = scaler.inverse_transform(
    prediction
)
print(
    "LSTM Prediction:",
    prediction[0][0]
)
