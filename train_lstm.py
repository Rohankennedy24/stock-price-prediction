import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load Dataset
df = pd.read_csv("dataset/cleaned_NVDA.csv")

# Use close price only
data = df[['close']].values

# Scale Data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)

# Create Sequences
X = []
y = []

sequence_length = 60

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i,0])
    y.append(scaled_data[i,0])

X = np.array(X)
y = np.array(y)

# Reshape for LSTM
X = np.reshape(
    X,
    (X.shape[0], X.shape[1], 1)
)

print("X Shape:", X.shape)

# Build Model
model = Sequential()

model.add(
    LSTM(
        units=50,
        return_sequences=True,
        input_shape=(X.shape[1],1)
    )
)

model.add(
    LSTM(
        units=50
    )
)

model.add(
    Dense(25)
)

model.add(
    Dense(1)
)

# Compile
model.compile(
    optimizer='adam',
    loss='mean_squared_error'
)

# Train
model.fit(
    X,
    y,
    epochs=5,
    batch_size=32
)

# Save Model
model.save(
    "models/lstm_model.h5"
)

print("LSTM Model Saved!")
model.save("models/lstm_model.h5")