"""
01_sentiment_timeline.html - динамика sentiment + объём
02_tickers_breakdown.html - динамика по тикерам
03_anomalies.html -  аномальные значения сентимента
04_distribution.html - распределение сентимента
"""

import logging
import os
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from clean import MOEX_TICKERS

###############################################################################
# Конфигурация

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("viz")

COLORS = {
    "positive": "green",
    "negative": "red",
    "neutral": "grey",
    "primary": "blue",
}


COMMON_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Arial, sans-serif", size=12),
    title_font=dict(size=18, family="Arial, sans-serif"),
    margin=dict(l=60, r=40, t=80, b=60),
    hoverlabel=dict(bgcolor="white", font_size=12),
)

# Параметры для rolling
ROLLING_WINDOW = 30
MIN_PERIODS = 10
Z_THRESHOLD = 2.0

###############################################################################
# Подготовка данных
def daily_aggregate(sentiment):
    df = sentiment.copy()
    df["day"] = pd.to_datetime(df["date"]).dt.date
    
    daily = df.groupby("day").agg(
        n=("sentiment_score", "size"),
        mean_score=("sentiment_score", "mean"),
        n_pos=("sentiment_label", lambda x: (x == "positive").sum()),
        n_neg=("sentiment_label", lambda x: (x == "negative").sum())).reset_index()
    
    daily["day"] = pd.to_datetime(daily["day"])
    daily["net_sentiment"] = (daily["n_pos"] - daily["n_neg"]) / daily["n"].clip(lower=1)
    
    # Z-статистика
    rolling_mean = daily["mean_score"].rolling(ROLLING_WINDOW, min_periods=MIN_PERIODS).mean()
    rolling_std = daily["mean_score"].rolling(ROLLING_WINDOW, min_periods=MIN_PERIODS).std()
    daily["zscore"] = (daily["mean_score"] - rolling_mean) / rolling_std
    
    return daily

###############################################################################
# 1. Sentiment timeline
def plot_sentiment_timeline(daily, out_path):
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=(
            "Sentiment Net Score: (позитив − негатив) / всего постов",
            "Объём публикаций по дням",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=daily["day"], y=daily["net_sentiment"],
            mode="lines",
            name="Net Sentiment",
            line=dict(color=COLORS["primary"], width=2),
            fill="tozeroy",
            fillcolor="rgba(46, 134, 171, 0.2)",
            hovertemplate="%{x|%Y-%m-%d}<br>Net: %{y:.3f}<extra></extra>",
        ),
        row=1, col=1,
    )

    fig.add_hline(
        y=0, line=dict(color="black", width=1, dash="dash"),
        row=1, col=1,
    )

    fig.add_trace(
        go.Bar(
            x=daily["day"], y=daily["n"],
            name="Posts/day",
            marker=dict(color=COLORS["primary"], opacity=0.6),
            hovertemplate="%{x|%Y-%m-%d}<br>Posts: %{y}<extra></extra>",
        ),
        row=2, col=1,
    )

    fig.update_yaxes(title_text="Net Sentiment", row=1, col=1)
    fig.update_yaxes(title_text="Posts count", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)

    fig.update_layout(
        height=700,
        title="MarketTwits — Sentiment Timeline",
        showlegend=False,
        **COMMON_LAYOUT,
    )

    fig.write_html(out_path, include_plotlyjs="cdn")
    logger.info(f"→ {out_path}")


###############################################################################
# 2. Динамика по тикерам

def plot_tickers_breakdown(posts, sentiment, out_path, top_n=15):
    df = posts.merge(sentiment[["id", "sentiment_score"]], on="id")
    df = df.explode("tickers").dropna(subset=["tickers"])
    df = df[df["tickers"].isin(MOEX_TICKERS)]

    if df.empty:
        logger.warning("Нет данных по тикерам MOEX_TICKERS - пропуск графика")
        return

    ticker_stats = (
        df.groupby("tickers")
        .agg(
            mentions=("sentiment_score", "size"),
            mean_sent=("sentiment_score", "mean"),
        )
        .reset_index()
        .sort_values("mentions", ascending=False)
        .head(top_n)
        .sort_values("mentions")
    )

    colors = [
        COLORS["negative"] if s < -0.1
        else COLORS["positive"] if s > 0.1
        else COLORS["neutral"]
        for s in ticker_stats["mean_sent"]
    ]

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.5, 0.5],
        horizontal_spacing=0.15,
        subplot_titles=(
            f"Топ-{top_n} тикеров MOEX по упоминаниям",
            "Средний sentiment по тикерам",
        ),
    )

    fig.add_trace(
        go.Bar(
            x=ticker_stats["mentions"],
            y=ticker_stats["tickers"],
            orientation="h",
            marker=dict(color=COLORS["primary"]),
            text=ticker_stats["mentions"],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Mentions: %{x}<extra></extra>",
            showlegend=False,
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Bar(
            x=ticker_stats["mean_sent"],
            y=ticker_stats["tickers"],
            orientation="h",
            marker=dict(color=colors),
            text=[f"{s:+.2f}" for s in ticker_stats["mean_sent"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Sentiment: %{x:.3f}<extra></extra>",
            showlegend=False,
        ),
        row=1, col=2,
    )

    fig.add_vline(x=0, line=dict(color="black", width=1, dash="dash"), row=1, col=2)

    fig.update_xaxes(title_text="Number of mentions", row=1, col=1)
    fig.update_xaxes(title_text="Mean sentiment score", row=1, col=2)

    fig.update_layout(
        height=600,
        title=f"Топ-{top_n} тикеров MOEX - упоминания и sentiment",
        **COMMON_LAYOUT,
    )

    fig.write_html(out_path, include_plotlyjs="cdn")
    logger.info(f"→ {out_path}")


###############################################################################
# 3. Аномалии

def plot_anomalies(daily, out_path, z_threshold=2.0):
    fig = go.Figure()

    colors = [
        COLORS["negative"] if z < -z_threshold
        else COLORS["positive"] if z > z_threshold
        else COLORS["neutral"]
        for z in daily["zscore"].fillna(0)
    ]

    fig.add_trace(
        go.Bar(
            x=daily["day"], y=daily["zscore"],
            name="Z-score",
            marker=dict(color=colors, opacity=0.85),
            hovertemplate="%{x|%Y-%m-%d}<br>Z-score: %{y:.2f}<extra></extra>",
        )
    )

    fig.add_hline(
        y=z_threshold, line=dict(color="darkred", width=1, dash="dash"),
        annotation_text=f"+{z_threshold}σ", annotation_position="right",
    )
    fig.add_hline(
        y=-z_threshold, line=dict(color="darkred", width=1, dash="dash"),
        annotation_text=f"−{z_threshold}σ", annotation_position="right",
    )
    fig.add_hline(y=0, line=dict(color="black", width=1))

    anomalies = daily[abs(daily["zscore"]) > z_threshold].copy()
    for _, row in anomalies.iterrows():
        fig.add_annotation(
            x=row["day"], y=row["zscore"],
            text=row["day"].strftime("%m-%d"),
            showarrow=True, arrowhead=2,
            ax=0, ay=-30 if row["zscore"] > 0 else 30,
            font=dict(size=10),
        )

    fig.update_layout(
        height=500,
        title=f"Аномалии sentiment (z-score > {z_threshold})",
        xaxis_title="Date",
        yaxis_title="Z-score",
        showlegend=False,
        **COMMON_LAYOUT,
    )

    fig.write_html(out_path, include_plotlyjs="cdn")
    logger.info(f"→ {out_path}")


###############################################################################
# 4. Распределение

def plot_distribution(sentiment, out_path):
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.6, 0.4],
        horizontal_spacing=0.15,
        subplot_titles=(
            "Распределение sentiment_score",
            "Доли классов",
        ),
        specs=[[{"type": "xy"}, {"type": "domain"}]],
    )

    for label, color in [
        ("negative", COLORS["negative"]),
        ("neutral", COLORS["neutral"]),
        ("positive", COLORS["positive"]),
    ]:
        scores = sentiment[sentiment["sentiment_label"] == label]["sentiment_score"]
        if len(scores) > 0:
            fig.add_trace(
                go.Histogram(
                    x=scores, name=label.capitalize(),
                    marker=dict(color=color, opacity=0.7),
                    nbinsx=40,
                    hovertemplate=f"<b>{label}</b><br>Score: %{{x:.2f}}<br>Count: %{{y}}<extra></extra>",
                ),
                row=1, col=1,
            )

    label_counts = sentiment["sentiment_label"].value_counts()
    fig.add_trace(
        go.Pie(
            labels=label_counts.index,
            values=label_counts.values,
            marker=dict(colors=[
                COLORS["negative"] if l == "negative"
                else COLORS["positive"] if l == "positive"
                else COLORS["neutral"]
                for l in label_counts.index
            ]),
            textinfo="label+percent",
            hole=0.4,
        ),
        row=1, col=2,
    )

    fig.update_xaxes(title_text="Sentiment score", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)

    fig.update_layout(
        height=500,
        title="Распределение sentiment_score",
        barmode="overlay",
        showlegend=True,
        **COMMON_LAYOUT,
    )

    fig.write_html(out_path, include_plotlyjs="cdn")
    logger.info(f"{out_path}")


###############################################################################
# Main
def main():
    out_dir = Path(os.getenv("OUTPUT_DIR", "./output"))
    out_dir.mkdir(exist_ok=True)

    posts = pd.read_parquet(out_dir / "posts_clean.parquet")
    sentiment = pd.read_parquet(out_dir / "sentiment.parquet")

    logger.info(f"Posts: {len(posts)}, Sentiment: {len(sentiment)}")

    daily = daily_aggregate(sentiment)
    daily.to_csv(out_dir / "daily_sentiment.csv", index=False)
    logger.info(f"Daily aggregate: {len(daily)} дней")

    plot_sentiment_timeline(daily, out_dir / "01_sentiment_timeline.html")
    plot_tickers_breakdown(posts, sentiment, out_dir / "02_tickers_breakdown.html")
    plot_anomalies(daily, out_dir / "03_anomalies.html")
    plot_distribution(sentiment, out_dir / "04_distribution.html")

###############################################################################
if __name__ == "__main__":
    main()