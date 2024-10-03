from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
from preprocess import preprocess_text

# Load trained model
with open('models/candidate_ranking_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/rank', methods=['POST'])
def rank():
    data = request.json
    candidates = pd.DataFrame(data['candidates'])
    ranked_candidates = rank_candidates(candidates)
    return jsonify(ranked_candidates.to_dict(orient='records'))

def rank_candidates(candidates):
    candidates['processed_response'] = candidates['response'].apply(preprocess_text)
    scores = model.predict(candidates['processed_response'])
    candidates['score'] = scores
    ranked_candidates = candidates.sort_values(by='score', ascending=False)
    return ranked_candidates

if __name__ == '__main__':
    app.run(debug=True)
