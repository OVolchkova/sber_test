import logging
import re
from pathlib import Path
from datetime import timedelta
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("signals")

###############################################################################
# Ключевые слова, сигнализирующие о рисках
RISK_WORDS = {
    "санкции", "санкций", "санкциям", "санкциями",
    "штраф", "штрафа", "штрафов", "штрафы",
    "расследование", "проверка", "проверки",
    "банкротство", "дефолт",
    "риск", "уголовное", "обыск", "арест",
    "приостановка", "приостановлен",
    "отзыв лицензии", "ликвидация",
    "сокращение", "увольнение",
    "падение", "обвал", "понижение",
    "downgrade", "негатив"
}

RISK_REGEX = re.compile("|".join(RISK_WORDS), re.IGNORECASE)

###############################################################################
def find_topic_bursts(posts, clusters, method, min_posts=3, max_avg_per_day=1.0):
    df = posts.merge(clusters, on="id")
    df["date_only"] = pd.to_datetime(df["date"]).dt.date
    daily_counts = df.groupby(["cluster", "date_only"]).size().reset_index(name="posts_count")
    avg_freq = daily_counts.groupby("cluster")["posts_count"].mean()
    rare_clusters = avg_freq[avg_freq <= max_avg_per_day].index
    
    # Отбираем всплески у редких кластеров
    bursts = daily_counts[(daily_counts["cluster"].isin(rare_clusters)) & (daily_counts["posts_count"] >= min_posts)]
    
    signals = []
    for _, row in bursts.iterrows():
        cluster = row["cluster"]
        day = row["date_only"]
        posts_in_burst = df[(df["cluster"] == cluster) & (df["date_only"] == day)]
        all_tickers = set()
        for t in posts_in_burst["tickers"]:
            all_tickers.update(t)
        
        signals.append({
            "type": "topic_burst",
            "date": day,
            "cluster": cluster,
            "method": method,
            "post_id": posts_in_burst["id"].tolist(), # список ID
            "ticker": ",".join(sorted(all_tickers)),
            "reason": f"{row['posts_count']} постов в редком кластере {cluster}",
            "text": posts_in_burst.iloc[0]["text_clean"],
        })
    
    return pd.DataFrame(signals)

###############################################################################
# Ищет посты, где одновременно упоминается тикер и слова риска
def find_risk_mentions(posts):
    with_tickers = posts[posts["tickers"].apply(len) > 0].copy()
    if with_tickers.empty:
        return pd.DataFrame()
    
    # Ищем риск-слова
    with_tickers["risk_found"] = with_tickers["text_clean"].str.lower().apply(lambda x: RISK_REGEX.findall(x))
    risky = with_tickers[with_tickers["risk_found"].apply(len) > 0]
    
    signals = []
    for i, row in risky.iterrows():
        signals.append({
            "type": "ticker_risk",
            "date": pd.to_datetime(row["date"]).date(),
            "cluster": -1,
            "method": "lexicon",
            "post_id": row["id"],
            "ticker": ",".join(row["tickers"]),
            "reason": f"тикеры {','.join(row['tickers'])} + риск: {row['risk_found'][:3]}",
        })
    
    return pd.DataFrame(signals)

###############################################################################
def find_sentiment_shifts(posts, sentiment, lookback_days=30, recent_days=3, min_shift=0.5):
    df = posts.merge(sentiment[["id", "sentiment_score"]], on="id")
    df["date_only"] = pd.to_datetime(df["date"]).dt.date
    exploded = df.explode("tickers").dropna(subset=["tickers"])
    last_day = exploded["date_only"].max()
    old_start = last_day - timedelta(days=lookback_days)
    old_end = last_day - timedelta(days=recent_days)
    new_start = last_day - timedelta(days=recent_days)
    
    signals = []
    for ticker, group in exploded.groupby("tickers"):
        old = group[(group["date_only"] >= old_start) & (group["date_only"] < old_end)]["sentiment_score"]
        new_period = group[group["date_only"] >= new_start]
        new = new_period["sentiment_score"]
        if len(old) < 2 or len(new) < 1:
            continue
        
        old_mean = old.mean()
        new_mean = new.mean()
        if pd.isna(old_mean) or pd.isna(new_mean):
            continue
        
        shift = new_mean - old_mean
        if abs(shift) < min_shift:
            continue

        last_post = new_period.iloc[-1]
        
        direction = "позитив" if shift > 0 else "негатив"
        signals.append({
            "type": "sentiment_shift",
            "date": last_day,
            "cluster": -1,
            "method": "sentiment",
            "post_id": last_post["id"],
            "ticker": ticker,
            "reason": f"{direction} сдвиг {shift:+.2f} (старое {old_mean:.2f} - новое {new_mean:.2f})",
            "text": last_post["text_clean"],
        })
    
    return pd.DataFrame(signals)

###############################################################################
def main():
    data_dir = Path("output")
    posts = pd.read_parquet(data_dir / "posts_clean.parquet")
    sentiment = pd.read_parquet(data_dir / "sentiment.parquet")
    
    all_signals = []
    
    # 1. Всплески редких кластеров
    for clusters_file in data_dir.glob("clusters_*.csv"):
        method = clusters_file.stem.replace("clusters_", "")
        clusters = pd.read_csv(clusters_file)
        
        bursts = find_topic_bursts(posts, clusters, method, min_posts=3)
        if not bursts.empty:
            all_signals.append(bursts)
            logger.info(f"{method}: {len(bursts)} всплесков тем")
    
    # 2. Риск + тикеры
    risks = find_risk_mentions(posts)
    if not risks.empty:
        all_signals.append(risks)
        logger.info(f"риск+тикер: {len(risks)}")
    
    # 3. Резкие смены сентимента
    shifts = find_sentiment_shifts(posts, sentiment, min_shift=0.5)
    if not shifts.empty:
        all_signals.append(shifts)
        logger.info(f"сдвиг сентимента: {len(shifts)}")
    
    if not all_signals:
        logger.warning("Сигналов не найдено")
        return
    
    result = pd.concat(all_signals, ignore_index=True)
    result = result.sort_values(["date", "type"]).reset_index(drop=True)
    result.to_csv(data_dir / "signals.csv", index=False, encoding="utf-8-sig")

###############################################################################
if __name__ == "__main__":
    main()