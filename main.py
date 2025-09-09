from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
from google_play_scraper import search, reviews

# ---- NLTK Setup ----
nltk.data.path.append("nltk_data")
vader_analyzer = SentimentIntensityAnalyzer()

# ---- RoBERTa Setup ----
roberta = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

# ---- Flask App ----
app = Flask(__name__)
CORS(app)

# ---- Helper Functions ----
def analyze_vader(text):
    cleaned = text.replace("\n", " ").strip()
    return vader_analyzer.polarity_scores(cleaned)

def analyze_roberta(text):
    result = roberta(text)[0]   # single dict
    label_map = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}
    label = label_map.get(result['label'], "Neutral")
    score = result['score']

    # Apply threshold rule
    if score <= 0.55:
        sentiment = "Neutral"
    else:
        sentiment = label

    return {
        "sentiment": sentiment,
        "confidence": round(score, 3),
        "label": label  # original label for reference if needed
    }

def pick_example_review(roberta_scores, review_texts, roberta_overall, roberta_threshold=0.55):
    filtered = [
        (i, r) for i, r in enumerate(roberta_scores)
        if r['sentiment'] == roberta_overall and r['confidence'] >= roberta_threshold
    ]
    if filtered:
        matching_index = max(filtered, key=lambda x: x[1]['confidence'])[0]
    else:
        matching_index = max(enumerate(roberta_scores), key=lambda x: x[1]['confidence'])[0]

    return review_texts[matching_index], roberta_scores[matching_index]['confidence']

def aggregate_sentiment(review_texts, roberta_threshold=0.7):
    if not review_texts:
        return {
            "vader": {"overall": "Neutral", "avg_compound": 0},
            "roberta": {"sentiment": "Neutral", "confidence": 0},
            "example_review": "",
            "example_confidence": 0,
            "sentiment_counts": {"Positive": 0, "Neutral": 0, "Negative": 0},
            "note": "No reviews found for this app."
        }

    vader_scores = [analyze_vader(t) for t in review_texts]
    avg_vader = sum([v['compound'] for v in vader_scores]) / len(vader_scores)
    vader_overall = "Positive" if avg_vader >= 0.05 else "Negative" if avg_vader <= -0.05 else "Neutral"

    roberta_scores = [analyze_roberta(t) for t in review_texts]
    sentiment_counts = {"Positive": 0, "Neutral": 0, "Negative": 0}

    for r in roberta_scores:
        if r['confidence'] < roberta_threshold:
            sentiment_counts["Neutral"] += 1
        else:
            sentiment_counts[r['sentiment']] += 1

    roberta_overall = max(sentiment_counts, key=sentiment_counts.get)
    roberta_confidence = sentiment_counts[roberta_overall] / len(roberta_scores)

    matching_review, matching_confidence = pick_example_review(roberta_scores, review_texts, roberta_overall, roberta_threshold)

    return {
        "vader": {"overall": vader_overall, "avg_compound": avg_vader},
        "roberta": {"sentiment": roberta_overall, "confidence": round(roberta_confidence, 2)},
        "example_review": matching_review,
        "example_confidence": round(matching_confidence, 2),
        "sentiment_counts": sentiment_counts
}

def fetch_reviews(app_name, count=50):
    search_results = search(app_name, lang="en", country="us")
    if not search_results:
        return []
    package_id = search_results[0]['appId']
    fetched_reviews, _ = reviews(package_id, lang='en', country='us', count=count)
    return [r['content'] for r in fetched_reviews if r.get('content')]

# ---- Routes ----
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    app_name = data.get("text", "").strip()
    if not app_name:
        return jsonify({"error": "No app name provided"}), 400

    reviews = fetch_reviews(app_name)
    if not reviews:
        return jsonify({"error": f"No reviews found for {app_name}"}), 404

    result = aggregate_sentiment(reviews)
    return jsonify(result)

@app.route('/single')
def single_page():
    return render_template('single.html')

@app.route('/compare_page')
def compare_page():
    return render_template('compare.html')


@app.route('/compare', methods=['POST'])
def compare():
    data = request.get_json()
    app1 = data.get("app1", "").strip()
    app2 = data.get("app2", "").strip()

    if not app1 or not app2:
        return jsonify({"error": "Both app names are required"}), 400

    reviews1 = fetch_reviews(app1)
    reviews2 = fetch_reviews(app2)

    if not reviews1 or not reviews2:
        return jsonify({"error": "One or both apps have no reviews"}), 404

    result1 = aggregate_sentiment(reviews1)
    result2 = aggregate_sentiment(reviews2)

    # Quick pros/cons extraction using top reviews
    def extract_pros_cons(reviews):
        pros = []
        cons = []
        for r in reviews:
            score = analyze_vader(r)['compound']
            rob = analyze_roberta(r)
            if score >= 0.05 or rob['sentiment'] == "Positive":
                pros.append(r)
            elif score <= -0.05 or rob['sentiment'] == "Negative":
                cons.append(r)
        return {
            "pros": pros[:3] if pros else ["No strong positives detected"],
            "cons": cons[:3] if cons else ["No strong negatives detected"]
        }

    pros_cons1 = extract_pros_cons(reviews1)
    pros_cons2 = extract_pros_cons(reviews2)

    return jsonify({
        "app1": {"name": app1, **result1, **pros_cons1},
        "app2": {"name": app2, **result2, **pros_cons2}
    })


def compare_apps():
    data = request.get_json()
    app1_name = data.get('app1')
    app2_name = data.get('app2')

    # Fetch and analyze reviews for both apps
    app1_result = analyze_app(app1_name)  # Should return sentiment, example review, pros/cons, counts
    app2_result = analyze_app(app2_name)

    return jsonify({
        'app1': app1_result,
        'app2': app2_result
    })    

if __name__ == "__main__":
    app.run(debug=True)
