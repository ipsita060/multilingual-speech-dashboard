import speech_recognition as sr
from textblob import TextBlob

# Initialize recognizer
recognizer = sr.Recognizer()

# Use microphone
with sr.Microphone() as source:
    print(" Speak something...")
    audio = recognizer.listen(source)

try:
    # Convert speech to text
    text = recognizer.recognize_google(audio)
    print("\n Transcribed Text:", text)

    # NLP - Sentiment Analysis
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity

    if sentiment > 0:
        print(" Sentiment: Positive")
    elif sentiment < 0:
        print(" Sentiment: Negative")
    else:
        print(" Sentiment: Neutral")

except sr.UnknownValueError:
    print("Could not understand audio")
except sr.RequestError:
    print("API error")