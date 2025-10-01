#!/usr/bin/env python3
"""
ethical_advisor.py
Lightweight AI Ethics Advisor prototype:
- Accepts text (model description / code snippet) or dataset CSV and returns ethical analysis + recommendations.
- CLI and simple Flask POST endpoint for interactive use.

Author: (starter template) — extend as needed.
"""

import os
import json
import sys
import time
import argparse
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List

# Core libs
import pandas as pd
import numpy as np

# NLP & ML (guarded)
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None

try:
    from transformers import pipeline
    summarizer = pipeline("summarization", truncation=True)
    zero_shot = pipeline("zero-shot-classification")
except Exception:
    summarizer = None
    zero_shot = None

# Optional fairness toolkits (guarded)
try:
    import aif360
    from aif360.datasets import StandardDataset
    from aif360.metrics import BinaryLabelDatasetMetric
    AIF360_AVAILABLE = True
except Exception:
    AIF360_AVAILABLE = False

# Simple logger / audit trail
AUDIT_DIR = "ethics_audit"
os.makedirs(AUDIT_DIR, exist_ok=True)


# ----------------------
# Utilities
# ----------------------
def now_iso():
    return datetime.utcnow().isoformat() + "Z"


def save_audit(audit: Dict[str, Any], name: Optional[str] = None) -> str:
    if name is None:
        name = f"audit_{int(time.time())}.json"
    path = os.path.join(AUDIT_DIR, name)
    with open(path, "w", encoding="utf8") as f:
        json.dump(audit, f, indent=2)
    return path


# ----------------------
# Analysis modules
# ----------------------
SENSITIVE_KEYWORDS = [
    "gender", "sex", "race", "ethnicity", "age", "birth", "religion",
    "health", "medical", "location", "address", "ssn", "social security",
    "income", "salary", "credit card", "payment", "phone", "email"
]


def detect_sensitive_mentions(text: str) -> List[str]:
    found = set()
    txt = text.lower()
    for kw in SENSITIVE_KEYWORDS:
        if kw in txt:
            found.add(kw)
    # use spaCy entities if available
    if nlp is not None:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in ("PERSON", "GPE", "LOC", "NORP", "ORG", "DATE"):
                found.add(ent.label_)
    return sorted(found)


def summarize_text(text: str, max_len: int = 200) -> str:
    text = text.strip()
    if len(text) == 0:
        return ""
    if summarizer is not None:
        try:
            out = summarizer(text, max_length=60, min_length=20, do_sample=False)
            return out[0]["summary_text"]
        except Exception:
            pass
    # fallback naive summarizer: first 2 sentences
    sentences = text.split(".")
    return (". ".join([s.strip() for s in sentences[:2] if s.strip()]) + ".").strip()


def code_static_checks(code: str) -> List[str]:
    issues = []
    txt = code.lower()
    if "password" in txt or "secret" in txt or "api_key" in txt or "authorization" in txt:
        issues.append("Hard-coded secrets or API keys detected. Use a secrets manager or environment variables.")
    if "requests.post(" in txt and "https://" in txt and "token" in txt:
        issues.append("Network calls with tokens appear in code — avoid embedding secrets or transmit securely.")
    if "print(" in txt and ("ssn" in txt or "email" in txt):
        issues.append("Code may print PII to stdout — avoid logging sensitive info.")
    if "open(" in txt and ".write" in txt and ("password" in txt or "secret" in txt):
        issues.append("Writing sensitive data to disk in plain text detected.")
    # add more heuristics as needed
    return issues


# ----------------------
# Dataset checks
# ----------------------
def basic_dataset_checks(df: pd.DataFrame, label_col: Optional[str] = None, protected_attrs: Optional[List[str]] = None) -> Dict[str, Any]:
    out = {}
    out["rows"] = len(df)
    out["columns"] = df.shape[1]
    out["missing_values"] = df.isnull().sum().to_dict()
    out["sample_rows"] = df.head(3).to_dict(orient="records")
    # label distribution
    if label_col and label_col in df.columns:
        out["label_distribution"] = df[label_col].value_counts(dropna=False).to_dict()
    # protected attributes checks
    detected_protected = []
    if not protected_attrs:
        # guess protected attributes heuristically
        cand = [c for c in df.columns if any(k in c.lower() for k in ["gender", "sex", "race", "age", "ethnic", "religion", "country", "zip", "location"])]
        protected_attrs = cand
    for attr in protected_attrs:
        if attr in df.columns:
            detected_protected.append(attr)
            vals = df[attr].value_counts(dropna=False).to_dict()
            out[f"protected_{attr}_counts"] = vals
            # show imbalance ratio for top two classes when binary-ish
            try:
                series = df[attr].dropna()
                if series.nunique() <= 10:
                    vc = series.value_counts(normalize=True)
                    out[f"protected_{attr}_imbalance"] = vc.to_dict()
            except Exception:
                pass
    out["detected_protected_attributes"] = detected_protected
    return out


# ----------------------
# Fairness checks (optional using aif360)
# ----------------------
def aif360_fairness_check(df: pd.DataFrame, label_col: str, protected_attribute: str) -> Dict[str, Any]:
    if not AIF360_AVAILABLE:
        return {"error": "AIF360 not installed or not available. Install it to compute detailed fairness metrics."}
    # aif360 requires specific formatting; this is a simple example for binary labels
    try:
        # try to create StandardDataset
        dataset = StandardDataset(
            df,
            label_name=label_col,
            favorable_classes=[1],
            protected_attribute_names=[protected_attribute],
            privileged_classes=[[1]]
        )
        metric = BinaryLabelDatasetMetric(dataset, privileged_groups=[{protected_attribute: 1}], unprivileged_groups=[{protected_attribute: 0}])
        di = metric.disparate_impact()
        mean_diff = metric.mean_difference()
        return {"disparate_impact": di, "mean_difference": mean_diff}
    except Exception as e:
        return {"error": f"AIF360 check failed: {str(e)}"}


# ----------------------
# High-level analysis pipeline
# ----------------------
def analyze_text_input(text: str) -> Dict[str, Any]:
    result = {
        "timestamp": now_iso(),
        "type": "text",
        "raw_text": text[:4000],  # truncate stored raw for audit
    }
    result["summary"] = summarize_text(text)
    result["sensitive_mentions"] = detect_sensitive_mentions(text)
    # quick risk scoring (heuristic)
    score = 0
    if len(result["sensitive_mentions"]) > 0:
        score += 30
    if "training" in text.lower() and "biased" in text.lower():
        score += 10
    if "personal data" in text.lower() or "pii" in text.lower():
        score += 20
    result["heuristic_risk_score_0_100"] = min(100, score)
    # recommendations
    recs = []
    if "gender" in result["sensitive_mentions"] or "race" in result["sensitive_mentions"]:
        recs.append("Check fairness across groups; consider disaggregated metrics and use fairness-aware algorithms.")
    if "health" in result["sensitive_mentions"] or "medical" in text.lower():
        recs.append("Medical data is sensitive — ensure consent, proper de-identification, and regulatory compliance.")
    if "pii" in text.lower() or "personal data" in text.lower():
        recs.append("Perform a privacy impact assessment, remove direct identifiers, and consider differential privacy.")
    if not recs:
        recs.append("No obvious sensitive keywords detected; still validate with dataset-level checks if data is used.")
    result["recommendations"] = recs
    return result


def analyze_code_input(code: str) -> Dict[str, Any]:
    result = {
        "timestamp": now_iso(),
        "type": "code",
        "issues": code_static_checks(code),
        "sensitive_mentions": detect_sensitive_mentions(code),
        "summary": summarize_text(code)
    }
    # simple remediation suggestions
    rem = []
    if any("secret" in s.lower() or "api key" in s.lower() for s in result["issues"]):
        rem.append("Remove secrets from code; use environment variables or dedicated secret store.")
    if "PII" in code.upper() or "ssn" in code.lower():
        rem.append("Mask or remove PII fields; add logging controls so PII isn't written to logs.")
    if not rem:
        rem.append("Run static analysis and code review focused on security and privacy.")
    result["recommendations"] = rem
    return result


def analyze_dataset(path: str, label_col: Optional[str] = None, protected_attrs: Optional[List[str]] = None) -> Dict[str, Any]:
    audit = {"timestamp": now_iso(), "type": "dataset", "path": path}
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return {"error": f"Failed to read dataset: {e}"}
    checks = basic_dataset_checks(df, label_col=label_col, protected_attrs=protected_attrs)
    audit["checks"] = checks
    # optional fairness metrics: pick first detected protected attribute
    if AIF360_AVAILABLE and checks["detected_protected_attributes"]:
        try:
            protected = checks["detected_protected_attributes"][0]
            fairness = aif360_fairness_check(df, label_col or "label", protected)
            audit["aif360"] = fairness
        except Exception as e:
            audit["aif360_error"] = str(e)
    else:
        if not AIF360_AVAILABLE:
            audit["aif360"] = "AIF360 not installed; install for deeper fairness metrics."
    # privacy heuristics
    pii_likely = [c for c in df.columns if any(k in c.lower() for k in ["name", "email", "phone", "ssn", "address", "zip"])]
    audit["likely_pii_columns"] = pii_likely
    # quick recommendations
    recs = []
    if checks["rows"] < 500:
        recs.append("Dataset is small (<500 rows). Consider more data or robust cross-validation and uncertainty estimation.")
    if pii_likely:
        recs.append("PII-like columns detected. Remove or pseudonymize, and check legal compliance.")
    if checks.get("detected_protected_attributes"):
        recs.append("Evaluate model performance per protected group; consider re-sampling or fairness-aware algorithms.")
    audit["recommendations"] = recs
    return audit


# ----------------------
# Conversational / Q&A (simple)
# ----------------------
def answer_question(question: str, analysis_summary: Dict[str, Any]) -> str:
    q = question.lower()
    # simple rule-based routing
    if "risk" in q or "score" in q:
        return f"Heuristic risk score: {analysis_summary.get('heuristic_risk_score_0_100','N/A')}. See recommendations: {analysis_summary.get('recommendations')}"
    if "sensitive" in q or "pii" in q:
        return f"Detected sensitive mentions: {analysis_summary.get('sensitive_mentions')}. If you supplied a dataset, run dataset checks to find PII columns."
    if "fair" in q or "fairness" in q:
        return "Fairness: check model performance across protected groups. Consider IBM AIF360 for detailed metrics. See recommendations: " + ", ".join(analysis_summary.get("recommendations", []))
    # fallback: summarize
    return f"Summary: {analysis_summary.get('summary','No summary available')}\nRecommendations: {analysis_summary.get('recommendations','No recommendations available')}"


# ----------------------
# CLI / HTTP server
# ----------------------
def run_cli():
    p = argparse.ArgumentParser(prog="ethical_advisor.py", description="AI Ethics Advisor CLI")
    p.add_argument("--input-text", help="Model description or text to analyze")
    p.add_argument("--input-code", help="Path to code file to analyze")
    p.add_argument("--input-dataset", help="Path to CSV dataset to analyze")
    p.add_argument("--label-col", help="Label column name for dataset (optional)")
    p.add_argument("--protected", nargs="*", help="Protected attribute column names (optional)")
    p.add_argument("--save-audit", action="store_true", help="Save audit JSON to disk")
    args = p.parse_args()

    result = {}
    if args.input_text:
        result = analyze_text_input(open(args.input_text).read() if os.path.isfile(args.input_text) else args.input_text)
    elif args.input_code:
        result = analyze_code_input(open(args.input_code, "r", encoding="utf8").read())
    elif args.input_dataset:
        result = analyze_dataset(args.input_dataset, label_col=args.label_col, protected_attrs=args.protected)
    else:
        print("No input supplied. Use --input-text or --input-code or --input-dataset.")
        sys.exit(1)

    print("\n--- Ethics Advisor Report ---\n")
    print(json.dumps(result, indent=2))
    if args.save_audit:
        path = save_audit(result)
        print(f"\nAudit saved to {path}")


# Simple Flask API for POSTing inputs
def create_flask_app():
    from flask import Flask, request, jsonify
    app = Flask(__name__)

    @app.route("/analyze", methods=["POST"])
    def analyze():
        try:
            payload = request.get_json(force=True)
            kind = payload.get("type")
            if kind == "text":
                txt = payload.get("text","")
                res = analyze_text_input(txt)
            elif kind == "code":
                code = payload.get("code","")
                res = analyze_code_input(code)
            elif kind == "dataset":
                path = payload.get("path")
                if not path:
                    return jsonify({"error":"dataset analysis expects a 'path' pointing to a CSV file accessible by the server."}), 400
                res = analyze_dataset(path, label_col=payload.get("label_col"), protected_attrs=payload.get("protected"))
            else:
                return jsonify({"error":"unknown type; valid types: text, code, dataset"}), 400
            # save audit automatically
            audit_name = f"audit_{int(time.time())}.json"
            save_audit(res, name=audit_name)
            return jsonify({"result": res, "audit_file": audit_name})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    @app.route("/qa", methods=["POST"])
    def qa():
        payload = request.get_json(force=True)
        question = payload.get("question","")
        # we will try to load the latest audit for context
        audits = sorted([f for f in os.listdir(AUDIT_DIR) if f.endswith(".json")], reverse=True)
        if not audits:
            return jsonify({"answer": "No existing analysis in audit history. Run /analyze first."})
        latest = os.path.join(AUDIT_DIR, audits[0])
        with open(latest, "r", encoding="utf8") as f:
            summary = json.load(f)
        ans = answer_question(question, summary)
        return jsonify({"answer": ans, "audit_used": audits[0]})

    return app


if __name__ == "__main__":
    # simple dispatch: if invoked with --serve, run flask
    if "--serve" in sys.argv:
        app = create_flask_app()
        print("Starting Ethics Advisor server on http://127.0.0.1:5000")
        app.run(debug=False, host="0.0.0.0", port=5000)
    else:
        run_cli()
