Ultimate Real-Time News Sentiment Dashboard
==============================================

Overview
------------
Ultimate Real-Time News Sentiment Dashboard is an AI-powered Streamlit web application that fetches and analyzes live news articles from multiple trusted APIs — NewsAPI, GNews, and CurrentsAPI.

It uses NLP and Sentiment Analysis to classify news headlines as Positive, Negative, or Neutral using a hybrid combination of VADER, RoBERTa, and FinBERT models. The dashboard includes interactive analytics, maps, and word clouds for deep insight into news sentiment trends.
Key Features
----------------
- Real-time news from multiple APIs
- Hybrid Sentiment Analysis (VADER + RoBERTa + FinBERT)
- Interactive Streamlit Dashboard
- Region, Category, and Sentiment Filtering
- Geographical Sentiment Mapping
- Keyword Extraction via TF-IDF
- Article Clustering using K-Means
- Word Clouds for Positive and Negative News
- CSV Export Functionality

Tech Stack
--------------
- Frontend: Streamlit
- Backend: Python
- AI Models: VADER, RoBERTa, FinBERT
- Libraries: Transformers, Pandas, Plotly, WordCloud, Scikit-learn, PyCountry, Matplotlib

Installation and Setup
--------------------------
1. Clone the repository:
   git clone https://github.com/<your-username>/news-sentiment-dashboard.git
   cd news-sentiment-dashboard

2. Install dependencies:
   pip install -r requirements.txt

3. Add your API keys inside app.py:
   NEWSAPI_KEY = "your_newsapi_key"
   GNEWS_KEY = "your_gnews_key"
   CURRENTS_KEY = "your_currentsapi_key"

4. Run the application:
   streamlit run app3.py

5. Open the displayed local URL (usually http://localhost:8501).

📊 Visual Insights
-------------------
- Sentiment Trend (Moving Average)
- Sentiment Distribution (Pie Chart)
- Word Clouds (Positive/Negative News)
- Country-wise Sentiment Map
- Keyword Extraction using TF-IDF
- Article Clustering Visualization

Example Use Cases
---------------------
- Media Analytics and Monitoring
- Financial Market Sentiment Tracking
- Brand Reputation Analysis
- Research and Academic Studies

