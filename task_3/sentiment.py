import logging
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

###############################################################################
# Конфигурация 
###############################################################################

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sentiment")

###############################################################################
# Положительные и отрицательные корни
POS_LEX = {
    "рост", "выросл", "рекорд", "прибыль", "дивиденд", "ралли",
    "опереди", "превзо", "позитив", "buy", "покупк",
}
NEG_LEX = {
    "падени", "упал", "обвал", "убыт", "санкц", "штраф",
    "дефолт", "банкрот", "негатив", "продаж", "sell", "downgrade",
    "приостанов", "снижени", "сокращ",
}

LABEL_MAP = {
    "negative": -1.0,
    "neutral": 0.0,
    "positive": 1.0,
}

###############################################################################
# Загрузка модели

MODEL_NAME = "seara/rubert-tiny2-russian-sentiment"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = (AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device))
model.eval()
id2label = {int(k): v for k, v in model.config.id2label.items()}

###############################################################################
# Предсказание модели
###############################################################################

@torch.inference_mode()
def predict_sentiment(texts, batch_size=32):

    labels = []
    scores = []

    for i in tqdm(range(0, len(texts), batch_size), desc="sentiment"):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=256,  return_tensors="pt").to(device)
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

        for p in probs:
            idx = int(np.argmax(p))
            label = id2label[idx].lower()
            labels.append(label)
            weights = np.array([LABEL_MAP.get(id2label[j].lower(), 0.0) for j in range(len(p))])
            scores.append(float(np.dot(p, weights)))

    return labels, np.array(scores)

###############################################################################
# Лексикон
def lexicon_score(text):
    t = text.lower()
    pos = sum(1 for w in POS_LEX if w in t)
    neg = sum(1 for w in NEG_LEX if w in t)
    if pos == neg == 0:
        return 0.0

    return (pos - neg) / (pos + neg)

###############################################################################
# Комбинация модели и словаря
# Мдель обучена на обычных русских текстах, поэтмоу чтобы скорректировать её под финансы, надо добавить оценку лексикона
# Примерно на 30% возьмем оценку из лексикона, на 70 - из модели

def combine(model_score, lex_score, alpha=0.7):
    score = alpha * model_score + (1 - alpha) * lex_score
    return float(np.clip(score, -1, 1))

###############################################################################
# у меня не было размеченных данных, поэтмоу я не откалибровала 0.2
# 0.2 я посчитала нормальным порогом, так как он не будет давать слишком много false negative и снизится полнота
def score_to_label(score, threshold=0.2):
    if score > threshold:
        return "positive"
    if score < -threshold:
        return "negative"

    return "neutral"

###############################################################################
# Pipeline

def run(input_parquet, output_parquet):
    df = pd.read_parquet(input_parquet)
    labels, model_scores = predict_sentiment(df["text_clean"].tolist())
    lex_scores = df["text_clean"].map(lexicon_score).values
    final_scores = np.array([combine(m, l) for m, l in zip(model_scores, lex_scores)])
    final_labels = np.array([score_to_label(s) for s in final_scores])

    out = pd.DataFrame({
        "id": df["id"].values,
        "date": df["date"].values,
        "model_label": labels,
        "model_score": model_scores,
        "lexicon_score": lex_scores,
        "sentiment_score": final_scores,
        "sentiment_label": final_labels,
    })

    out.to_parquet(output_parquet, index=False)

    logger.info(
        f"Распределение: "
        f"{dict(pd.Series(final_labels).value_counts())}"
    )

    logger.info(f"{output_parquet}")

###############################################################################

if __name__ == "__main__":
    out_dir = Path(os.getenv("OUTPUT_DIR", "./output"))
    run(out_dir / "posts_clean.parquet", out_dir / "sentiment.parquet")