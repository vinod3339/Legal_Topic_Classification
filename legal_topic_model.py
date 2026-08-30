"""
Legal Topic Model (LTM) Implementation
Embedding-based clustering topic model for legal document analysis.
"""

import os
import json
import re
import torch
import numpy as np
import joblib
from collections import Counter
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer, AutoModel

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

DEFAULT_DOMAIN_SEEDS = {
    "Criminal Law": "offence trial conviction acquittal sentence bail evidence murder theft assault culpable homicide IPC CrPC fine summons warrant investigation FIR confession charge sheet imprisonment accused prosecution defense police cross-examination criminal appeal parole remand cognizable non-cognizable crime scene witness testimony mens rea",
    "Civil Procedure": "plaint written statement injunction affidavit decree execution jurisdiction CPC stay hearing summons interlocutory order revision appeal limitation decree-holder judgment-debtor res judicata interim relief order 39 rule 1 rule 2 temporary injunction cross suit cause of action maintainability",
    "Constitutional Law": "fundamental rights article equality writ petition judicial review public interest litigation PIL mandamus certiorari habeas corpus federalism directive principles preamble constitution amendment separation of powers judiciary legislature executive constitutional validity freedom of speech religion right to life natural justice due process",
    "Contract Law": "agreement breach consideration damages indemnity guarantee performance termination offer acceptance specific relief clause liquidated damages force majeure contract formation repudiation anticipatory breach novation assignment agency void voidable contract coercion undue influence misrepresentation fraud remedy estoppel",
    "Property Law": "ownership possession transfer lease mortgage tenancy title sale deed landlord eviction partition easement right of way immovable property movable property conveyance gift trust intestate succession registration land revenue mutation specific performance transfer of property act adverse possession co-ownership",
    "Family Law": "marriage divorce custody adoption maintenance guardianship succession domestic violence Hindu Marriage Act Muslim Law special marriage act annulment legitimacy inheritance dowry cruelty separation alimony spouse child support family court",
    "Labour Law": "employment wages dismissal retrenchment industrial dispute trade union reinstatement gratuity employee compensation minimum wages collective bargaining standing orders layoff termination provident fund social security employee state insurance strike lockout conciliation labour court award industrial tribunal",
    "Tax Law": "income tax GST assessment penalty deduction refund exemption appeal tribunal TDS audit reassessment interest excise customs input tax credit registration turnover advance ruling export import valuation anti-profiteering tax evasion surcharge cess direct tax indirect tax tax liability",
    "Intellectual Property": "patent trademark copyright design infringement licensing royalties intellectual property trade secret passing off GI intellectual property rights registration renewal opposition invalidation counterfeit piracy domain dispute IPAB WIPO novelty inventive step originality moral rights fair use",
    "Company Law": "corporate directors shareholders insolvency merger takeover securities SEBI corporate governance NCLT Memorandum Articles winding up incorporation prospectus share capital dividend AGM minority protection oppression mismanagement board resolution company secretary audit compliance corporate social responsibility",
    "Consumer Law": "consumer deficiency service compensation unfair trade practice refund product liability warranty advertisement consumer protection act defect goods negligence district forum national commission unfair terms misleading representation class action recall deceptive conduct",
    "Environmental Law": "pollution clearance EIA waste management forest conservation NGT environmental protection water air act wildlife biodiversity climate change emission hazardous substance conservation act environment impact assessment sustainable development ecological balance public nuisance green tribunal",
    "Arbitration and ADR": "arbitration conciliation mediation award agreement tribunal dispute resolution foreign award enforcement arbitration act seat venue arbitrator appointment arbitral proceedings jurisdiction interim relief settlement institutional arbitration UNCITRAL model law arbitration clause",
    "Evidence Law": "admissibility relevancy proof exhibit cross-examination expert witness burden of proof documentary evidence primary evidence secondary evidence confession dying declaration hearsay res gestae presumption oral evidence electronic record forensic chain of custody circumstantial evidence corroboration",
    "Land Acquisition": "compensation land acquisition fair compensation resettlement eminent domain market value notification public purpose rehabilitation land acquisition act award possession section 11 notification section 6 notification land use consent farmer",
    "Insolvency and Bankruptcy": "corporate debtor resolution professional moratorium NCLT IBC resolution plan committee of creditors CIRP liquidation insolvency resolution code insolvency professional secured creditor debt restructuring voluntary liquidation avoidance transaction",
    "Service Law": "recruitment promotion seniority disciplinary action transfer retirement pension departmental inquiry suspension reinstatement public servant misconduct service rules government servant pay scale leave rules administrative tribunal",
    "Cyber Law": "information technology act data protection privacy cybercrime hacking phishing identity theft intermediary liability digital signature electronic contract computer virus cyber security encryption cyber terrorism cyber forensics",
    "Banking and Finance Law": "loan recovery NPA RBI regulation SARFAESI negotiable instruments cheque dishonour banking ombudsman secured creditor mortgage hypothecation repo agreement guarantee letter of credit financial institution debt recovery tribunal DRT",
    "Real Estate Law": "builder buyer agreement possession delay RERA real estate project promoter allottee occupancy certificate construction flat registration title dispute defect warranty refund",
    "Administrative Law": "delegated legislation administrative discretion natural justice quasi-judicial tribunal review inquiry regulation ordinance statutory authority abuse of power ultra vires show cause notice"
}


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    return a_norm @ b_norm.T


class LegalTopicModelLTM:
    """Legal Topic Model (LTM) class for clustering and topic prediction."""

    def __init__(self, ltm_path: str = 'ltm_model_DEPLOYMENT_READY.joblib', model_name: str = "law-ai/InLegalBERT"):
        """
        Initialize the Legal Topic Model.
        
        Args:
            ltm_path: Path to the saved LTM model file
            model_name: Base transformer model name
        """
        self.ltm_path = ltm_path
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = None
        self.model = None
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_ltm_path = ltm_path if os.path.isabs(ltm_path) else os.path.normpath(os.path.join(base_dir, ltm_path))
        
        if os.path.exists(full_ltm_path):
            print(f"Loading existing LTM model from {full_ltm_path}...")
            data = joblib.load(full_ltm_path)
            self.topic_centroids_ = data["topic_centroids_"]
            self.topic_map = data["topic_map"]
            self.topic_keywords_ = data["topic_keywords_"]
            self.model_name = data.get("model_name", model_name)
        else:
            print(f"Initializing new LTM model and saving to {full_ltm_path}...")
            self._initialize_from_seeds(full_ltm_path)

    def _ensure_model_loaded(self):
        if self.tokenizer is None or self.model is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            local_fallback = os.path.join(base_dir, 'inlegalbert_finetuned', 'final_model')
            if os.path.exists(local_fallback):
                model_path_to_use = local_fallback
            else:
                model_path_to_use = self.model_name
                
            self.tokenizer = AutoTokenizer.from_pretrained(model_path_to_use)
            self.model = AutoModel.from_pretrained(model_path_to_use).to(self.device)
            self.model.eval()

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        self._ensure_model_loaded()
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
                embeddings.append(emb)
        return np.array(embeddings)

    def _initialize_from_seeds(self, save_path: str):
        topic_names = list(DEFAULT_DOMAIN_SEEDS.keys())
        seed_texts = list(DEFAULT_DOMAIN_SEEDS.values())
        
        centroids = self._encode_texts(seed_texts)
        
        self.topic_centroids_ = centroids
        self.topic_map = {i: name for i, name in enumerate(topic_names)}
        self.topic_keywords_ = {}
        
        for i, (name, text) in enumerate(DEFAULT_DOMAIN_SEEDS.items()):
            words = text.split()
            kw_list = [f"{w} ({round(0.95 - idx*0.05, 4)})" for idx, w in enumerate(words[:12])]
            self.topic_keywords_[i] = kw_list
            
        stable_components = {
            "topic_centroids_": self.topic_centroids_,
            "topic_map": self.topic_map,
            "topic_keywords_": self.topic_keywords_,
            "model_name": self.model_name
        }
        
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        joblib.dump(stable_components, save_path)
        print(f"Saved LTM model components to {save_path}")

    def chunk_text(self, text: str, max_words: int = 250, overlap: int = 50) -> List[str]:
        words = text.split()
        if len(words) <= max_words:
            return [text]
        
        chunks = []
        step = max_words - overlap
        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i + max_words])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    def predict(self, text: str) -> Dict:
        """
        Predict topic for the given text.
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary with topic prediction results.
        """
        if not text or not text.strip():
            raise ValueError("Missing or empty 'text' field.")
            
        chunks = self.chunk_text(text)
        chunk_embs = self._encode_texts(chunks)
        
        sims = cosine_sim(chunk_embs, self.topic_centroids_)
        
        chunk_distribution = {}
        top_topic_indices = np.argmax(sims, axis=1)
        
        for idx in top_topic_indices:
            label = self.topic_map[int(idx)]
            chunk_distribution[label] = chunk_distribution.get(label, 0) + 1
            
        doc_avg_sim = sims.mean(axis=0)
        dominant_topic_idx = int(np.argmax(doc_avg_sim))
        predicted_label = self.topic_map[dominant_topic_idx]
        
        raw_topic_kws = self.topic_keywords_.get(dominant_topic_idx, [])
        topic_keywords = [str(kw) for kw in raw_topic_kws]
        
        doc_keywords = self._extract_doc_keywords(text)
        
        return {
            "predicted_label": predicted_label,
            "topic_keywords": topic_keywords,
            "document_keywords": doc_keywords,
            "chunk_distribution": chunk_distribution,
            "total_chunks": len(chunks)
        }

    def _extract_doc_keywords(self, text: str, top_n: int = 10) -> List[str]:
        if HAS_SKLEARN:
            try:
                vectorizer = TfidfVectorizer(stop_words='english', max_features=top_n)
                tfidf_matrix = vectorizer.fit_transform([text])
                feature_names = vectorizer.get_feature_names_out()
                scores = tfidf_matrix.toarray()[0]
                
                sorted_indices = np.argsort(scores)[::-1]
                doc_kws = [f"{feature_names[i]} ({round(float(scores[i]), 4)})" for i in sorted_indices if scores[i] > 0]
                if doc_kws:
                    return doc_kws
            except Exception:
                pass
                
        # Pure Python fallback for document keyword extraction
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'over', 'after', 'is', 'are', 'was', 'were',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'shall', 'should', 'may', 'might', 'must', 'can', 'could', 'this', 'that', 'these',
            'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who', 'whom',
            'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
            'too', 'very', 's', 't', 'just', 'don', 'now', 'under', 'filed', 'petitioner', 'respondent'
        }
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        filtered = [w for w in words if w not in stop_words]
        counts = Counter(filtered).most_common(top_n)
        
        if not counts:
            return ["legal (0.5000)", "document (0.4000)"]
            
        total = sum(c for _, c in counts)
        return [f"{word} ({round(count / total, 4)})" for word, count in counts]
