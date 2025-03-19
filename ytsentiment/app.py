import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from googleapiclient.discovery import build
from nltk.sentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv

# Load API Key from .env file
load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")

# Function to extract video ID from different YouTube URL formats
def extract_video_id(url):
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]  # For standard YouTube URLs
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]  # For shortened YouTube URLs
    else:
        raise ValueError("Invalid YouTube URL format")

# Function to get YouTube comments
def get_youtube_comments(video_id):
    youtube = build("youtube", "v3", developerKey=API_KEY)
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=50  # Can increase as needed
    )
    response = request.execute()

    comments = [item["snippet"]["topLevelComment"]["snippet"]["textDisplay"] for item in response.get("items", [])]
    return comments

# Function for Sentiment Analysis
def analyze_sentiments(comments):
    sia = SentimentIntensityAnalyzer()
    sentiment_results = []
    
    for comment in comments:
        score = sia.polarity_scores(comment)
        if score['compound'] >= 0.05:
            sentiment_results.append("Positive")
        elif score['compound'] <= -0.05:
            sentiment_results.append("Negative")
        else:
            sentiment_results.append("Neutral")

    return sentiment_results

# Run sentiment analysis on a YouTube video
if __name__ == "__main__":
    video_url = input("Enter YouTube Video URL: ")

    try:
        video_id = extract_video_id(video_url)  # Extract video ID
        print("Extracted Video ID:", video_id)  # Debugging output

        comments = get_youtube_comments(video_id)
        if not comments:
            print("No comments found for this video.")
            exit()

        sentiments = analyze_sentiments(comments)

        df = pd.DataFrame({"Comment": comments, "Sentiment": sentiments})
        sentiment_counts = df["Sentiment"].value_counts(normalize=True) * 100

        # Plot results
        plt.figure(figsize=(5, 5))
        sns.set_style("darkgrid")
        sentiment_counts.plot(kind="pie", autopct="%.1f%%", colors=["green", "red", "blue"])
        plt.title("Sentiment Analysis of YouTube Comments")
        plt.show()

    except ValueError as e:
        print("Error:", e)
    except Exception as e:
        print("An unexpected error occurred:", e)
