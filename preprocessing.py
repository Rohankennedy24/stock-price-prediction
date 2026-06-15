import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv("dataset/NVDA.csv")

print(df.head())
print("\nDataset Shape:")

print(df.shape)
print("\nDataset Information")

print(df.info())
print("\nMissing Values")
print(df.isnull().sum())

df = df.dropna()
print("\nDuplicate Rows:")
print(df.duplicated().sum())

df = df.drop_duplicates()
df['Date'] = pd.to_datetime(df['Date'])

print(df.dtypes)
print(df.describe())


plt.figure(figsize=(12,6))
plt.plot(df['close'])
plt.title("NVIDIA Closing Price")
plt.xlabel("Days")
plt.ylabel("Price")
plt.show()

X = df[['open','high','low','volume']]
y = df['close']

print(X.head())
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(X_train.shape)
print(X_test.shape)

df.to_csv(
    "dataset/cleaned_NVDA.csv",
    index=False
)
print("Dataset cleaned and saved.")

df.drop('ingested_at_utc', axis=1, inplace=True)
print(df.columns)

X = df[['open','high','low','volume']]
y = df['close']

print("Features Shape:", X.shape)
print("Target Shape:", y.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)

df.to_csv("dataset/cleaned_NVDA.csv", index=False)
print("Cleaned dataset saved successfully!")