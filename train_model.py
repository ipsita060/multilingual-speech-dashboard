import os
import time
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import kagglehub

def download_and_train():
    print("🚀 Starting Model Training Process...")
    
    # 1. Download Dataset from Kaggle
    dataset_name = "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
    print(f"📥 Downloading Kaggle dataset: {dataset_name}")
    try:
        path = kagglehub.dataset_download(dataset_name)
        print("✅ Dataset downloaded successfully to:", path)
        
        # Locate the CSV file in the downloaded path
        csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
        if not csv_files:
            raise Exception("No CSV file found in the downloaded dataset.")
        
        dataset_path = os.path.join(path, csv_files[0])
    except Exception as e:
        print(f"⚠️ Could not download from Kaggle ({e}).")
        print("Fallback: Please ensure 'dataset.csv' is present in the current directory.")
        dataset_path = "dataset.csv"
        
        if not os.path.exists(dataset_path):
            print("❌ dataset.csv not found! Creating a dummy dataset for demonstration instead.")
            # Create a small dummy dataset so it doesn't crash completely
            dummy_data = {
                'review': ['I loved this movie, it was fantastic!', 'Terrible movie, waste of time.', 
                           'Absolutely wonderful and inspiring.', 'Boring and dull.', 
                           'The acting was great, highly recommend.', 'Worst experience ever.'],
                'sentiment': ['positive', 'negative', 'positive', 'negative', 'positive', 'negative']
            }
            pd.DataFrame(dummy_data).to_csv(dataset_path, index=False)
            
    # 2. Load the Dataset
    print("📊 Loading dataset...")
    df = pd.read_csv(dataset_path)
    
    # Take a sample if dataset is too large to speed up local training during testing
    if len(df) > 10000:
        print("🔪 Sampling 10,000 rows for faster training...")
        df = df.sample(10000, random_state=42)
        
    print(f"Dataset Shape: {df.shape}")
    
    # Assume standard IMDB columns: 'review' and 'sentiment'
    if 'review' not in df.columns or 'sentiment' not in df.columns:
        print("⚠️ Warning: Could not find 'review' and 'sentiment' columns. Using first two columns instead.")
        df.columns = ['review', 'sentiment'] + list(df.columns[2:])
        
    # Map sentiment to binary 1 or 0
    df['label'] = df['sentiment'].apply(lambda x: 1 if str(x).lower().strip() == 'positive' else 0)
    
    # 3. Preprocess and Vectorize
    print("🧠 Vectorizing text data (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(df['review'])
    y = df['label']
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Train Model
    print("⚙️ Training Logistic Regression Classifier...")
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    
    # 5. Evaluate Accuracy
    print("📈 Evaluating Model...")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    
    print(f"\n✅ Model Training Complete!")
    print(f"🎯 Accuracy: {accuracy * 100:.2f}%")
    print(f"📜 Classification Report:\n{report}")
    
    # Save Model
    print("💾 Saving model files (model.pkl, vectorizer.pkl, metrics.pkl)...")
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
        
    # Save metrics so app.py can read them
    metrics = {
        "accuracy": float(accuracy),
        "dataset_size": len(df),
        "train_size": X_train.shape[0],
        "test_size": X_test.shape[0]
    }
    with open("metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)
        
    print("✨ All done! Ready for the Streamlit dashboard.")

if __name__ == "__main__":
    download_and_train()
