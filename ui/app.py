"""
Web Dashboard Application for Concept Drift Detection in Social Media Streams
Group 23:
- Eluri Manoj (2023BCD0029)
- P. Harshithvardhan (2023BCS0128)
- N. Lalith (2023BCD0033)

FastAPI Server providing REST endpoints and interactive UI to:
1. Replay Kafka-simulated streams window by window.
2. Run live inference and drift classification.
3. Visualize SDM trajectories, topic shifts, and comparative metrics in real time.
"""

import os
import sys
import json
import pandas as pd
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import preprocessor
from data.dataset_generator import DOMAIN_LEXICONS
from src.models.bert_sentiment import TSDABERTClassifier
from src.models.topic_model import StreamTopicModel
from src.drift.sentiment_drift import SentimentDriftDetector
from src.drift.topic_drift import TopicDriftDetector
from src.drift.domain_impact import DomainImpactScorer
from src.drift.multi_signal_detector import MultiSignalDriftDetector
from src.adaptation.model_adapter import AdaptiveModelUpdater

app = FastAPI(title="Concept Drift in Social Media Streams - Group 23")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
TEMPLATES_DIR = os.path.join(BASE_DIR, "ui", "templates")

templates = Jinja2Templates(directory=TEMPLATES_DIR)

# In-memory session state for active domain stream simulation
STATE = {
    'domain': 'manutd',
    'current_window_idx': 0,
    'total_windows': 6,
    'df': None,
    'model': None,
    'topic_model': None,
    'sent_detector': None,
    'topic_detector': None,
    'multi_detector': None,
    'dis_scorer': None,
    'adapter': None,
    'stream_history': []
}


def initialize_domain_session(domain: str):
    """Initializes models and detectors for the chosen domain."""
    csv_path = os.path.join(DATA_DIR, f"{domain}_tweets_stream.csv")
    if not os.path.exists(csv_path):
        from data.dataset_generator import generate_all_datasets
        generate_all_datasets(DATA_DIR)

    df = pd.read_csv(csv_path)
    w0_df = df[df['window_id'] == 'w0']
    t0_texts = w0_df['text'].tolist()
    t0_labels = w0_df['sentiment'].tolist()

    model = TSDABERTClassifier(use_fast_mode=True)
    model.fit_initial(t0_texts, t0_labels, epochs=12, lr=2e-3)

    topic_model = StreamTopicModel(n_topics=3)
    topic_model.fit(df['text'].tolist())

    sent_detector = SentimentDriftDetector(magnitude_threshold=0.35)
    topic_detector = TopicDriftDetector(jsd_threshold=0.20, cosine_threshold=0.25)
    multi_detector = MultiSignalDriftDetector(alpha=0.5, joint_threshold=0.30)

    lexicon = DOMAIN_LEXICONS.get(domain, set())
    dis_scorer = DomainImpactScorer(lexicon, threshold=0.35)
    adapter = AdaptiveModelUpdater(model, dis_scorer, max_buffer_size=300)
    adapter.initialize_buffer(t0_texts, t0_labels)

    # Initialize w0 baseline
    w0_preds, _, _ = model.predict(t0_texts)
    w0_sent_res = sent_detector.detect_drift(w0_preds, window_id="w0")
    w0_topic_dist, w0_centroid, w0_kw = topic_model.transform_window(t0_texts)
    w0_topic_res = topic_detector.detect_topic_drift(w0_topic_dist, w0_centroid, window_id="w0", keywords=w0_kw)
    w0_multi_res = multi_detector.evaluate_multi_signal(w0_sent_res, w0_topic_res, window_id="w0")

    STATE['domain'] = domain
    STATE['current_window_idx'] = 0
    STATE['df'] = df
    STATE['model'] = model
    STATE['topic_model'] = topic_model
    STATE['sent_detector'] = sent_detector
    STATE['topic_detector'] = topic_detector
    STATE['multi_detector'] = multi_detector
    STATE['dis_scorer'] = dis_scorer
    STATE['adapter'] = adapter
    STATE['stream_history'] = [{
        'window_id': 'w0',
        'sentiment_res': w0_sent_res,
        'topic_res': w0_topic_res,
        'multi_res': w0_multi_res,
        'accuracy': 1.0,
        'f1': 1.0,
        'selected_dis_count': 0
    }]


@app.on_event("startup")
def startup_event():
    initialize_domain_session("manutd")


@app.get("/", response_class=HTMLResponse)
def index_page(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "domain": STATE['domain'],
        "domains": ['manutd', 'stockmarket', 'covaxinwho']
    })


@app.post("/api/select_domain")
def api_select_domain(domain: str):
    if domain not in ['manutd', 'stockmarket', 'covaxinwho']:
        return JSONResponse({"error": "Invalid domain"}, status_code=400)
    initialize_domain_session(domain)
    return {"status": "ok", "domain": domain, "message": f"Initialized stream for #{domain}"}


@app.get("/api/state")
def api_get_state():
    return {
        "domain": STATE['domain'],
        "current_window_idx": STATE['current_window_idx'],
        "total_windows": 6,
        "history": STATE['stream_history']
    }


@app.post("/api/next_window")
def api_next_window():
    if STATE['current_window_idx'] >= 5:
        return {"status": "finished", "message": "Stream completed all 5 streaming windows"}

    next_idx = STATE['current_window_idx'] + 1
    w_id = f"w{next_idx}"
    w_df = STATE['df'][STATE['df']['window_id'] == w_id]
    if w_df.empty:
        return {"status": "error", "message": f"No data found for {w_id}"}

    texts = w_df['text'].tolist()
    labels = w_df['sentiment'].tolist()
    records = w_df.to_dict('records')

    # 1. Classification & Evaluation
    eval_res = STATE['model'].evaluate(texts, labels)
    preds, pos_probs, neg_probs = STATE['model'].predict(texts)

    # 2. Sentiment Drift (SDM)
    sent_res = STATE['sent_detector'].detect_drift(preds, window_id=w_id)

    # 3. Topic Drift (JSD / Cosine)
    topic_dist, centroid, kw = STATE['topic_model'].transform_window(texts)
    topic_res = STATE['topic_detector'].detect_topic_drift(topic_dist, centroid, window_id=w_id, keywords=kw)

    # 4. Multi-Signal Joint Detection
    multi_res = STATE['multi_detector'].evaluate_multi_signal(sent_res, topic_res, window_id=w_id)

    # 5. Model Adaptation using DIS
    adapt_res = STATE['adapter'].update_on_window(
        records,
        is_drift=multi_res['is_drift'],
        use_dis_filtering=True,
        window_id=w_id
    )

    sample_tweets = []
    for i in range(min(15, len(records))):
        dis, matched, _ = STATE['dis_scorer'].compute_dis(texts[i])
        sample_tweets.append({
            'text': texts[i],
            'true_sentiment': 'Positive' if labels[i] == 1 else 'Negative',
            'pred_sentiment': 'Positive' if preds[i] == 1 else 'Negative',
            'pos_prob': round(pos_probs[i], 3),
            'dis_score': round(dis, 3),
            'matched_words': list(matched)
        })

    window_summary = {
        'window_id': w_id,
        'window_drift_type': w_df['window_drift_type'].iloc[0],
        'accuracy': eval_res['accuracy'],
        'f1': eval_res['f1_score'],
        'sentiment_res': sent_res,
        'topic_res': topic_res,
        'multi_res': multi_res,
        'selected_dis_count': adapt_res['selected_tweets_count'],
        'sample_tweets': sample_tweets
    }

    STATE['current_window_idx'] = next_idx
    STATE['stream_history'].append(window_summary)

    return {
        "status": "ok",
        "window_summary": window_summary,
        "is_finished": next_idx >= 5
    }


@app.post("/api/reset")
def api_reset():
    initialize_domain_session(STATE['domain'])
    return {"status": "ok", "message": "Stream simulation reset to w0 baseline"}


@app.get("/api/benchmark_summary")
def api_get_benchmark_summary():
    report_path = os.path.join(RESULTS_DIR, "benchmark_report.json")
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            data = json.load(f)
        return data
    return {"error": "Benchmark report not generated yet"}


if __name__ == "__main__":
    import uvicorn
    print("Starting TSDA-BERT & Multi-Signal Concept Drift UI on http://localhost:8000")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
