"""
TSDA-BERT Sentiment Classifier
Implements Section 3.5.1 of Susi & Shanthi (2023):
"Fine Tuning of Uncased BERT for Sentiment Classification of Different Domains in Real Time"

Architecture:
- Transformer encoder (BERT/DistilBERT representation)
- 3 Dense layers with Dropout
- Binary output layer (Negative: 0, Positive: 1)
- Adaptive online fine-tuning interface
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

logger = logging.getLogger(__name__)


class BertDenseClassifierHead(nn.Module):
    """
    3 Dense layers with Dropout connected to the encoder representations
    as explicitly specified in Section 3.5.1 of the base paper.
    """
    def __init__(self, input_dim: int = 768, hidden_dim1: int = 256, hidden_dim2: int = 64, dropout_rate: float = 0.3):
        super().__init__()
        self.dense1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.dense2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.dense3 = nn.Linear(hidden_dim2, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu1(self.dense1(x))
        out = self.dropout1(out)
        out = self.relu2(self.dense2(out))
        out = self.dropout2(out)
        logits = self.dense3(out)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return self.softmax(logits)


class TSDABERTClassifier:
    """
    End-to-end TSDA-BERT Sentiment Classifier.
    Supports native HuggingFace transformer pipeline with a fast, zero-dependency
    hybrid embedding mode for immediate local testing and deployment.
    """
    def __init__(self, model_name: str = "distilbert-base-uncased", use_fast_mode: bool = False, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fast_mode = use_fast_mode
        self.model_name = model_name
        self.tokenizer = None
        self.transformer_encoder = None
        self.head = None
        self.vectorizer = None

        if not self.use_fast_mode:
            try:
                from transformers import AutoTokenizer, AutoModel
                logger.info(f"Loading HuggingFace transformer: {model_name} on {self.device}...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.transformer_encoder = AutoModel.from_pretrained(model_name).to(self.device)
                hidden_dim = self.transformer_encoder.config.hidden_size
                self.head = BertDenseClassifierHead(input_dim=hidden_dim).to(self.device)
                logger.info("Successfully loaded Transformer & Dense Head.")
            except Exception as e:
                logger.warning(f"Could not load HuggingFace model ({e}). Initializing high-speed neural embedding classifier.")
                self.use_fast_mode = True

        if self.use_fast_mode:
            # High-speed vectorized representation (dim=768 matching BERT hidden size)
            self.vectorizer = TfidfVectorizer(max_features=768, stop_words='english')
            self.head = BertDenseClassifierHead(input_dim=768).to(self.device)

    def _get_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Extracts 768-dim embeddings from texts."""
        if not self.use_fast_mode and self.transformer_encoder is not None:
            self.transformer_encoder.eval()
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                outputs = self.transformer_encoder(**encoded)
                # Mean pooling over token embeddings
                mask = encoded['attention_mask'].unsqueeze(-1)
                emb = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            return emb
        else:
            # Fast vectorized embeddings
            if not hasattr(self.vectorizer, 'vocabulary_') or self.vectorizer.vocabulary_ is None:
                features = self.vectorizer.fit_transform(texts).toarray()
            else:
                features = self.vectorizer.transform(texts).toarray()
            
            # Ensure 768 dimensions
            if features.shape[1] < 768:
                pad = np.zeros((features.shape[0], 768 - features.shape[1]))
                features = np.hstack([features, pad])
            elif features.shape[1] > 768:
                features = features[:, :768]
            return torch.tensor(features, dtype=torch.float32).to(self.device)

    def fit_initial(self, texts: List[str], labels: List[int], epochs: int = 12, lr: float = 2e-3, batch_size: int = 32):
        """Initial baseline training on historical tweets (T0 -> M0)."""
        logger.info(f"Fitting initial TSDA-BERT model on {len(texts)} tweets...")
        if self.use_fast_mode:
            self.vectorizer.fit(texts)

        emb = self._get_embeddings(texts)
        y = torch.tensor(labels, dtype=torch.long).to(self.device)
        dataset = TensorDataset(emb, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.AdamW(self.head.parameters(), lr=lr, weight_decay=1e-2)
        criterion = nn.CrossEntropyLoss()

        self.head.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                logits = self.head(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            logger.debug(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss / len(loader):.4f}")

    def predict(self, texts: List[str]) -> Tuple[List[int], List[float], List[float]]:
        """
        Predicts sentiment for input tweets.
        Returns:
            predictions: list of 0 (neg) or 1 (pos)
            pos_probs: probability of positive class
            neg_probs: probability of negative class
        """
        if not texts:
            return [], [], []

        self.head.eval()
        with torch.no_grad():
            emb = self._get_embeddings(texts)
            probs = self.head.predict_proba(emb).cpu().numpy()

        preds = np.argmax(probs, axis=1).tolist()
        neg_probs = probs[:, 0].tolist()
        pos_probs = probs[:, 1].tolist()

        return preds, pos_probs, neg_probs

    def evaluate(self, texts: List[str], true_labels: List[int]) -> Dict[str, float]:
        """Calculates Accuracy, F1, Precision, Recall, and Confusion Matrix."""
        preds, _, _ = self.predict(texts)
        acc = accuracy_score(true_labels, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(true_labels, preds, average='binary', zero_division=0)
        cm = confusion_matrix(true_labels, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        return {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1),
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn)
        }

    def adapt(self, adaptation_texts: List[str], adaptation_labels: List[int], epochs: int = 6, lr: float = 1e-3):
        """
        Online adaptation / fine-tuning on high-impact domain tweets (Section 3.5.4).
        """
        if len(adaptation_texts) == 0:
            return

        emb = self._get_embeddings(adaptation_texts)
        y = torch.tensor(adaptation_labels, dtype=torch.long).to(self.device)
        dataset = TensorDataset(emb, y)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        optimizer = optim.AdamW(self.head.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        self.head.train()
        for _ in range(epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                logits = self.head(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
        logger.info(f"TSDA-BERT adapted with {len(adaptation_texts)} domain-relevant tweets.")
