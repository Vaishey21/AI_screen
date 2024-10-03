from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import pandas as pd
from preprocess import preprocess_data

def train_model():
    # Preprocess the data
    df = preprocess_data('data/interview_response.csv')
    
    # Features and Labels
    X = df['response']
    y = df[['technical_score', 'problem_solving_score', 'communication_score']].mean(axis=1)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create a pipeline with TF-IDF and a regression model
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('model', LinearRegression())
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Save the model
    with open('models/candidate_ranking_model.pkl', 'wb') as f:
        pickle.dump(pipeline, f)

if __name__ == "__main__":
    train_model()
