import pandas as pd
import pickle
from preprocess import preprocess_text

# Load trained model
with open('models/candidate_ranking_model.pkl', 'rb') as f:
    model = pickle.load(f)

def rank_candidates(candidates):
    # Preprocess the response text
    candidates['processed_response'] = candidates['response'].apply(preprocess_text)
    
    # Predict scores
    scores = model.predict(candidates['processed_response'])
    
    # Assign scores to candidates
    candidates['score'] = scores
    
    # Rank candidates based on their scores
    ranked_candidates = candidates.sort_values(by='score', ascending=False)
    
    return ranked_candidates[['candidate_id', 'score']]

# Example usage (for testing purposes)
if __name__ == "__main__":
    candidates = pd.DataFrame({
        'candidate_id': [5, 6],
        'response': [
            "I have experience deploying AI models for real-world problems.",
            "I am passionate about AI and learning new things."
        ]
    })
    
    ranked_candidates = rank_candidates(candidates)
    print(ranked_candidates)
