import requests
import pandas as pd
import re
from datetime import datetime, timedelta
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import streamlit as st
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pycountry
import time
from io import BytesIO
import base64
from newsapi import NewsApiClient

# -------------------------------
# CONFIG
# -------------------------------
NEWSAPI_KEY = ""
GNEWS_KEY = ""
CURRENTS_KEY = ""

vader = SentimentIntensityAnalyzer()
roberta = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")

newsapi = NewsApiClient(api_key=NEWSAPI_KEY)

# -------------------------------
# HELPERS
# -------------------------------
def clean_text(text):
    if not text:
        return ""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)
    return text.strip()

def hybrid_sentiment(text):
    if not text:
        return {"label": "Neutral", "score": 0.0}

    # 1. VADER
    vader_score = vader.polarity_scores(text)["compound"]
    vader_label = "Positive" if vader_score >= 0.05 else "Negative" if vader_score <= -0.05 else "Neutral"

    # 2. RoBERTa
    roberta_result = roberta(text[:512])[0]
    roberta_label = roberta_result["label"].capitalize()
    roberta_score = roberta_result["score"]

    # 3. FinBERT
    finbert_result = finbert(text[:512])[0]
    finbert_label = finbert_result["label"].capitalize()
    finbert_score = finbert_result["score"]

    # Normalize into a voting dictionary
    scores = {"Positive": 0, "Negative": 0, "Neutral": 0}

    # VADER weight
    scores[vader_label] += abs(vader_score)

    # RoBERTa weight
    scores[roberta_label] += roberta_score

    # FinBERT weight
    scores[finbert_label] += finbert_score

    # Final decision
    final_label = max(scores, key=scores.get)
    final_score = scores[final_label] / (sum(scores.values()) + 1e-6)

    return {
        "label": final_label,
        "score": round(final_score, 3),
        "debug": {
            "VADER": f"{vader_label} ({round(vader_score,3)})",
            "RoBERTa": f"{roberta_label} ({round(roberta_score,3)})",
            "FinBERT": f"{finbert_label} ({round(finbert_score,3)})"
        }
    }

def fetch_newsapi(query="India", category="general"):
    try:
        response = newsapi.get_top_headlines(q=query, category=category, language='en', page_size=20)
        return [
            {
                "title": art["title"],
                "description": art.get("description", ""),
                "source": art["source"]["name"],
                "url": art["url"],
                "publishedAt": art["publishedAt"]
            } for art in response.get("articles", [])
        ]
    except Exception as e:
        st.error(f"NewsAPI Error: {e}")
        return []

def fetch_gnews(query="India", max_results=10):
    url = f"https://gnews.io/api/v4/search?q={query}&lang=en&country=in&max={max_results}&apikey={GNEWS_KEY}"
    r = requests.get(url).json()
    return [
        {
            "title": art["title"],
            "description": art.get("description", ""),
            "source": art.get("author") or "GNews",
            "url": art["url"],
            "publishedAt": art["published"]
        } for art in r.get("news", [])
    ]

def fetch_currents(query="India"):
    url = f"https://api.currentsapi.services/v1/search?keywords={query}&language=en&apiKey={CURRENTS_KEY}"
    r = requests.get(url).json()
    return [
        {
            "title": art["title"],
            "description": art.get("description", ""),
            "source": art["author"] or "Currents",
            "url": art["url"],
            "publishedAt": art["published"]
        } for art in r.get("news", [])
    ]

def get_news_sentiment(query="India", category="general"):
    articles = []
    try: articles.extend(fetch_newsapi(query, category))
    except: pass
    try: articles.extend(fetch_gnews(query))
    except: pass
    try: articles.extend(fetch_currents(query))
    except: pass

    results = []
    for art in articles:
        text = clean_text(art["title"] + " " + art["description"])
        sentiment = hybrid_sentiment(text)
        results.append({
            "title": art["title"],
            "description": art.get("description", ""),
            "source": art["source"],
            "publishedAt": art["publishedAt"],
            "url": art["url"],
            "sentiment": sentiment["label"],
            "confidence": sentiment["score"],
            "debug": sentiment["debug"]
        })
    return pd.DataFrame(results)

def get_country_code(region_name):
    try:
        country = pycountry.countries.lookup(region_name)
        return country.alpha_3
    except:
        return None

def generate_csv_download_link(df):
    csv = df.to_csv(index=False).encode()
    b64 = base64.b64encode(csv).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="news_sentiment.csv">📥 Download Filtered Data as CSV</a>'

# -------------------------------
# STREAMLIT APP
# -------------------------------
st.set_page_config(page_title="Ultimate News Sentiment Dashboard", layout="wide")
st.title("🚀 Ultimate Real-time News Sentiment Analysis")

regions = st.multiselect("Select regions:", ["India", "USA", "UK", "Canada", "Australia", "Germany", "France", "Japan"], default=["India"])
categories = st.multiselect("Select categories:", ["business", "entertainment", "general", "health", "science", "sports", "technology"], default=["general"])

combined_df = pd.DataFrame()
with st.spinner("Fetching news, please wait..."):
    for reg in regions:
        for cat in categories:
            df_temp = get_news_sentiment(reg, cat)
            df_temp['region'] = reg
            df_temp['category'] = cat
            combined_df = pd.concat([combined_df, df_temp])

if combined_df.empty:
    st.warning("No news found for selected regions/categories.")
else:
    combined_df["publishedAt"] = pd.to_datetime(combined_df["publishedAt"], errors="coerce")
    combined_df = combined_df.dropna(subset=["publishedAt"])
    combined_df = combined_df.sort_values("publishedAt")

    sentiment_map = {"Positive": 1, "Neutral": 0, "Negative": -1}
    combined_df["sentiment_score"] = combined_df["sentiment"].map(sentiment_map)

    # -------------------------------
    # NEWS ARTICLES FIRST
    # -------------------------------
    st.subheader("📰 News Articles")

    # Advanced options
    search_term = st.text_input("🔍 Search in articles:", "")
    sort_option = st.selectbox("Sort articles by:", ["Newest First", "Oldest First", "Confidence Score"])
    sentiment_filter = st.radio("Filter by Sentiment:", ["All", "Positive", "Neutral", "Negative"], horizontal=True)

    # Apply search
    display_df = combined_df.copy()
    if search_term:
        display_df = display_df[display_df["title"].str.contains(search_term, case=False, na=False)]

    # Apply sentiment filter
    if sentiment_filter != "All":
        display_df = display_df[display_df["sentiment"] == sentiment_filter]

    # Apply sorting
    if sort_option == "Newest First":
        display_df = display_df.sort_values("publishedAt", ascending=False)
    elif sort_option == "Oldest First":
        display_df = display_df.sort_values("publishedAt", ascending=True)
    elif sort_option == "Confidence Score":
        display_df = display_df.sort_values("confidence", ascending=False)

    # Show articles
    if display_df.empty:
        st.warning("No articles match your filter/search.")
    else:
        for i, row in display_df.iterrows():
            sentiment_icon = "😀" if row['sentiment'] == "Positive" else "😐" if row['sentiment'] == "Neutral" else "😡"
            st.markdown(f"### {sentiment_icon} {row['title']}")
            st.write(f"📖 **Description:** {row.get('description','No details available.')}")
            st.write(f"📍 Region: {row['region']} | 🏷 Category: {row['category']} | 📰 Source: {row['source']}")
            st.write(f"🕒 Published: {row['publishedAt']} | Confidence: {round(row['confidence']*100,2)}%")
            st.write(f"**Sentiment:** {row['sentiment']} ({row['confidence']})")
            with st.expander("🔎 Model Breakdown"):
                st.json(row["debug"])
            st.write("---")

    # -------------------------------
    # ANALYTICS BELOW ARTICLES
    # -------------------------------
    st.subheader("📊 Sentiment Summary")
    total = len(combined_df)
    pos_pct = round(len(combined_df[combined_df["sentiment"]=="Positive"]) / total * 100, 2)
    neg_pct = round(len(combined_df[combined_df["sentiment"]=="Negative"]) / total * 100, 2)
    neu_pct = round(len(combined_df[combined_df["sentiment"]=="Neutral"]) / total * 100, 2)
    st.info(f"Positive: {pos_pct}% | Negative: {neg_pct}% | Neutral: {neu_pct}%")

    fig_pie = px.pie(combined_df, names="sentiment", title="Sentiment Distribution")
    st.plotly_chart(fig_pie, use_container_width=True)

    sentiment_counts = combined_df["sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ['sentiment', 'count']
    fig_bar = px.bar(sentiment_counts,
                     x="sentiment", y="count",
                     labels={"sentiment": "Sentiment", "count": "Count"},
                     title="Sentiment Count")
    st.plotly_chart(fig_bar, use_container_width=True)

    rolling_df = combined_df[['publishedAt', 'sentiment_score']].rolling('1h', on='publishedAt').mean().reset_index()
    fig_trend = px.line(
        rolling_df,
        x="publishedAt",
        y="sentiment_score",
        title="Sentiment Trend Forecast (1-hour Moving Average)",
        labels={"sentiment_score": "Sentiment Score"}
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    source_counts = combined_df["source"].value_counts().reset_index()
    source_counts.columns = ["Source", "Article Count"]
    fig_sources = px.bar(
        source_counts.head(10),
        x="Source", y="Article Count",
        title="Top 10 News Sources",
        labels={"Source": "News Source", "Article Count": "Number of Articles"}
    )
    st.plotly_chart(fig_sources, use_container_width=True)

    vectorizer = TfidfVectorizer(stop_words='english', max_features=20)
    tfidf_matrix = vectorizer.fit_transform(combined_df["title"].fillna(""))
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.sum(axis=0).A1
    tfidf_df = pd.DataFrame({"Keyword": feature_names, "Score": tfidf_scores}).sort_values(by="Score", ascending=False)

    fig_keywords = px.bar(
        tfidf_df,
        x="Keyword", y="Score",
        title="Top 20 Keywords via TF-IDF",
        labels={"Keyword": "Keyword", "Score": "TF-IDF Score"}
    )
    st.plotly_chart(fig_keywords, use_container_width=True)

    st.subheader("📚 Word Clouds")
    pos_text = " ".join(combined_df[combined_df["sentiment"] == "Positive"]["title"].tolist())
    neg_text = " ".join(combined_df[combined_df["sentiment"] == "Negative"]["title"].tolist())

    col1, col2 = st.columns([1,1])
    with col1:
        st.write("Positive News WordCloud")
        if pos_text.strip():
            wc_pos = WordCloud(width=400, height=300, background_color="white").generate(pos_text)
            plt.imshow(wc_pos, interpolation="bilinear")
            plt.axis("off")
            st.pyplot(plt)
        else:
            st.write("No positive words available.")

    with col2:
        st.write("Negative News WordCloud")
        if neg_text.strip():
            wc_neg = WordCloud(width=400, height=300, background_color="black", colormap="Reds").generate(neg_text)
            plt.imshow(wc_neg, interpolation="bilinear")
            plt.axis("off")
            st.pyplot(plt)
        else:
            st.write("No negative words available.")

    filter_sentiment = st.multiselect("Filter by Sentiment:", ["Positive", "Neutral", "Negative"], default=["Positive", "Neutral", "Negative"])
    filter_region = st.multiselect("Filter by Region:", regions, default=regions)
    filter_category = st.multiselect("Filter by Category:", ["business", "entertainment", "general", "health", "science", "sports", "technology"], default=["general"])

    filtered_df = combined_df[
        (combined_df["sentiment"].isin(filter_sentiment)) &
        (combined_df["region"].isin(filter_region)) &
        (combined_df["category"].isin(filter_category))
    ]

    combined_df['country_code'] = combined_df['region'].apply(get_country_code)
    fig_map = px.choropleth(
        combined_df,
        locations="country_code",
        color="sentiment_score",
        hover_name="region",
        color_continuous_scale="Plasma",
        title="🌍 Geolocation Sentiment Map"
    )
    st.plotly_chart(fig_map, use_container_width=True)

    fig_volume = px.histogram(
        filtered_df,
        x="publishedAt",
        nbins=50,
        title="News Volume Over Time",
        labels={"publishedAt": "Publication Time", "count": "Number of Articles"}
    )
    st.plotly_chart(fig_volume, use_container_width=True)

    avg_confidence = round(filtered_df["confidence"].mean() * 100, 2)
    fig_gauge = px.pie(
        names=["Average Confidence"],
        values=[avg_confidence],
        title=f"🤖 Average Sentiment Confidence: {avg_confidence}%"
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    if not filtered_df.empty:
        X = vectorizer.transform(filtered_df["title"].fillna(""))
        num_clusters = 5
        if X.shape[0] >= num_clusters:
            kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(X)
            filtered_df["cluster"] = kmeans.labels_

            fig_cluster = px.histogram(
                filtered_df,
                x="cluster",
                color="sentiment",
                title="🧱 Article Clusters")
