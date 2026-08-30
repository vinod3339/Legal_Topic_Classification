"""
Fine-tuned Legal Topic Model Classifier
Uses fine-tuned InLegalBERT model for sequence classification across legal domains.
"""

import os
import json
import torch
from typing import Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class FinetunedTopicModel:
    """Finetuned Topic Model class for legal text classification."""
    
    def __init__(self, model_path: str = './inlegalbert_finetuned/'):
        """
        Initialize the Fine-tuned Topic Model.
        
        Args:
            model_path: Path to the fine-tuned model directory
        """
        self.model_path = model_path
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.isabs(model_path):
            resolved_path = os.path.normpath(os.path.join(base_dir, model_path))
        else:
            resolved_path = model_path

        final_model_dir = os.path.join(resolved_path, 'final_model')
        if os.path.exists(final_model_dir):
            model_dir = final_model_dir
        else:
            model_dir = resolved_path
            
        id2label_file = os.path.join(resolved_path, 'id_to_label.json')
        if not os.path.exists(id2label_file):
            id2label_file = os.path.join(os.path.dirname(resolved_path), 'id_to_label.json')
            
        if os.path.exists(id2label_file):
            with open(id2label_file, 'r', encoding='utf-8') as f:
                raw_id2label = json.load(f)
                self.id2label = {int(k): v for k, v in raw_id2label.items()}
        else:
            self.id2label = {}

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.model.eval()

    def predict(self, text: str) -> Dict:
        """
        Predict topic for the given text using the fine-tuned model.
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary with predicted_label and confidence score.
        """
        if not text or not text.strip():
            raise ValueError("Missing or empty 'text' field.")
            
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]
            pred_idx = torch.argmax(probs).item()
            confidence = probs[pred_idx].item()
            
        label = self.id2label.get(pred_idx, self.model.config.id2label.get(str(pred_idx), f"Topic {pred_idx}"))
        
        return {
            "predicted_label": label,
            "confidence": round(float(confidence), 4)
        }
