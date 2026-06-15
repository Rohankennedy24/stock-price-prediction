import pandas as pd
import nltk

from nltk.sentiment import SentimentIntensityAnalyzer

# Download VADER dictionary (run first time only)
nltk.download('vader_lexicon')

# Load Dataset
df = pd.read_csv("dataset/cnbc_headlines.csv")

# Clean Column Names
df.columns = df.columns.str.strip()

print("Columns:")
print(df.columns)

# Create Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Function to classify sentiment
def analyze_sentiment(text):

    score = sia.polarity_scores(str(text))

    compound = score['compound']

    if compound >= 0.05:
        return "Positive"

    elif compound <= -0.05:
        return "Negative"

    else:
        return "Neutral"

# Sentiment Analysis on Headlines
df['Sentiment'] = df['Headlines'].apply(
    analyze_sentiment
)

# Display first 10 rows
print("\nSample Results:\n")

print(
    df[['Headlines','Sentiment']].head(10)
)

# Count sentiments
positive = (
    df['Sentiment']
    == 'Positive'
).sum()

negative = (
    df['Sentiment']
    == 'Negative'
).sum()

neutral = (
    df['Sentiment']
    == 'Neutral'
).sum()

print("\n--------------------------------")

print("Positive News :", positive)

print("Negative News :", negative)

print("Neutral News :", neutral)

print("--------------------------------")

# Overall Market Sentiment
if positive > negative:
    market_sentiment = "POSITIVE"

elif negative > positive:
    market_sentiment = "NEGATIVE"

else:
    market_sentiment = "NEUTRAL"

print(
    "\nOverall Market Sentiment :",
    market_sentiment
)

# Save Result
df.to_csv(
    "dataset/news_with_sentiment.csv",
    index=False
)

print(
    "\nSentiment Analysis Completed!"
)