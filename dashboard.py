import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import re
from collections import Counter
import requests
import time

# Text processing and ML libraries
try:
    from textblob import TextBlob
    from wordcloud import WordCloud
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.cluster import KMeans
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    from nltk.sentiment import SentimentIntensityAnalyzer
except ImportError as e:
    st.error(f"Missing required libraries: {e}")
    st.info("Please install: pip install textblob wordcloud scikit-learn nltk")

# Gemini API configuration
GEMINI_API_KEY = "AIzaSyDmPiDZGWkap0oKZzENszs9nsK-E-A2qMA"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"

class ReviewAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english') + stopwords.words('french'))
        
    def classify_sentiment(self, text):
        """Classify sentiment using VADER and TextBlob"""
        if not text or pd.isna(text):
            return "neutral"
        
        # VADER analysis
        vader_score = self.sia.polarity_scores(str(text))
        
        # TextBlob analysis
        blob = TextBlob(str(text))
        
        # Combined scoring
        combined_score = (vader_score['compound'] + blob.sentiment.polarity) / 2
        
        if combined_score >= 0.1:
            return "positive"
        elif combined_score <= -0.1:
            return "negative"
        else:
            return "neutral"
    
    def classify_user_type(self, job_title, employment_length, current_employee):
        """Classify user type based on job title and employment data"""
        if pd.isna(job_title):
            job_title = ""
        
        job_title = str(job_title).lower()
        
        # Intern classification
        intern_keywords = ['intern', 'stagiaire', 'stage', 'trainee', 'apprentice']
        if any(keyword in job_title for keyword in intern_keywords):
            return "Intern"
        
        # Senior/Manager classification
        senior_keywords = ['senior', 'lead', 'manager', 'director', 'head', 'principal', 'architect']
        if any(keyword in job_title for keyword in senior_keywords):
            return "Senior/Manager"
        
        # Consultant classification
        consultant_keywords = ['consultant', 'advisor', 'specialist']
        if any(keyword in job_title for keyword in consultant_keywords):
            return "Consultant"
        
        # Developer/Engineer classification
        dev_keywords = ['developer', 'engineer', 'programmer', 'analyst', 'scientist']
        if any(keyword in job_title for keyword in dev_keywords):
            return "Developer/Engineer"
        
        # Based on employment length
        if isinstance(employment_length, (int, float)):
            if employment_length <= 1:
                return "Junior/New Employee"
            elif employment_length >= 3:
                return "Experienced Employee"
        
        return "General Employee"
    
    def extract_year_from_date(self, date_str):
        """Extract year from various date formats"""
        if pd.isna(date_str):
            return None
        
        try:
            # Try different date formats
            date_str = str(date_str)
            
            # ISO format
            if 'T' in date_str:
                return datetime.fromisoformat(date_str.replace('T', ' ').split('.')[0]).year
            
            # French format (e.g., "4 août 2025")
            french_months = {
                'janvier': '01', 'février': '02', 'mars': '03', 'avril': '04',
                'mai': '05', 'juin': '06', 'juillet': '07', 'août': '08',
                'septembre': '09', 'octobre': '10', 'novembre': '11', 'décembre': '12'
            }
            
            for fr_month, num_month in french_months.items():
                if fr_month in date_str.lower():
                    parts = date_str.split()
                    if len(parts) >= 3:
                        return int(parts[-1])
            
            # Extract year using regex
            year_match = re.search(r'\b(20\d{2})\b', date_str)
            if year_match:
                return int(year_match.group(1))
                
        except Exception:
            pass
        
        return None
    
    def perform_topic_modeling(self, texts, n_topics=5):
        """Perform topic modeling using LDA"""
        if not texts or len(texts) < 2:
            return [], []
        
        # Clean and preprocess texts
        cleaned_texts = []
        for text in texts:
            if text and not pd.isna(text):
                # Remove special characters and convert to lowercase
                cleaned = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
                cleaned_texts.append(cleaned)
        
        if len(cleaned_texts) < 2:
            return [], []
        
        try:
            # Vectorize the texts
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2)
            )
            doc_term_matrix = vectorizer.fit_transform(cleaned_texts)
            
            # Perform LDA
            lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
            lda.fit(doc_term_matrix)
            
            # Extract topics
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            for topic_idx, topic in enumerate(lda.components_):
                top_words = [feature_names[i] for i in topic.argsort()[-10:]]
                topics.append(f"Topic {topic_idx + 1}: {', '.join(top_words[:5])}")
            
            return topics, lda.transform(doc_term_matrix)
        
        except Exception as e:
            st.error(f"Topic modeling error: {e}")
            return [], []

def call_gemini_api(prompt):
    """Call Gemini API for recommendations"""
    try:
        headers = {
            'Content-Type': 'application/json',
        }
        
        data = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }]
        }
        
        response = requests.post(GEMINI_URL, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                return result['candidates'][0]['content']['parts'][0]['text']
        
        return "Unable to generate recommendations at this time."
    
    except Exception as e:
        return f"Error generating recommendations: {str(e)}"

def load_and_process_data():
    """Load and process the merged reviews data"""
    try:
        with open('merged_reviews.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        reviews = data['reviews']
        
        # Initialize analyzer
        analyzer = ReviewAnalyzer()
        
        # Process each review
        processed_reviews = []
        for review in reviews:
            # Skip incomplete records
            if not isinstance(review, dict) or len(review) < 3:
                continue
                
            # Skip records that only have 'indexable' field
            if list(review.keys()) == ['indexable'] or len(review.keys()) <= 2:
                continue
            
            # Extract text for sentiment analysis
            text_content = ""
            if review.get('text'):
                if isinstance(review['text'], dict):
                    text_parts = []
                    for key, value in review['text'].items():
                        if value and not pd.isna(value) and str(value).strip():
                            text_parts.append(str(value))
                    text_content = " ".join(text_parts)
                else:
                    text_content = str(review['text'])
            
            # Add title to text content
            if review.get('title'):
                text_content = str(review['title']) + " " + text_content
            
            # Skip if no meaningful text content
            if not text_content.strip():
                continue
            
            # Classify sentiment
            sentiment = analyzer.classify_sentiment(text_content)
            
            # Classify user type
            user_type = analyzer.classify_user_type(
                review.get('job_title', ''),
                review.get('employment_length', 0),
                review.get('current_employee', None)
            )
            
            # Extract year
            year = analyzer.extract_year_from_date(review.get('submission_date', ''))
            
            processed_review = {
                'review_id': review.get('review_id', review.get('encryptedReviewId', '')),
                'source': review.get('source', 'unknown'),
                'title': review.get('title', ''),
                'text_content': text_content,
                'overall_rating': review.get('overall_rating', review.get('overallRating', 0)),
                'job_title': review.get('job_title', review.get('jobTitle', '')),
                'location': review.get('location', ''),
                'current_employee': review.get('current_employee', review.get('currentEmployee', None)),
                'employment_length': review.get('employment_length', 0),
                'submission_date': review.get('submission_date', review.get('submissionDate', '')),
                'sentiment': sentiment,
                'user_type': user_type,
                'year': year,
                'helpful_count': review.get('helpful_count', review.get('helpful', 0)),
                'raw_review': review
            }
            processed_reviews.append(processed_review)
        
        return pd.DataFrame(processed_reviews)
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def create_sentiment_charts(df):
    """Create sentiment analysis charts"""
    # Sentiment distribution
    sentiment_counts = df['sentiment'].value_counts()
    
    fig_pie = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="Overall Sentiment Distribution",
        color_discrete_map={
            'positive': '#2E8B57',
            'negative': '#DC143C',
            'neutral': '#FFD700'
        }
    )
    
    # Sentiment by user type
    sentiment_user = df.groupby(['user_type', 'sentiment']).size().reset_index(name='count')
    fig_bar = px.bar(
        sentiment_user,
        x='user_type',
        y='count',
        color='sentiment',
        title="Sentiment by User Type",
        color_discrete_map={
            'positive': '#2E8B57',
            'negative': '#DC143C',
            'neutral': '#FFD700'
        }
    )
    fig_bar.update_layout(xaxis_tickangle=45)
    
    # Sentiment over time
    if 'year' in df.columns and not df['year'].isna().all():
        yearly_sentiment = df.groupby(['year', 'sentiment']).size().reset_index(name='count')
        fig_line = px.line(
            yearly_sentiment,
            x='year',
            y='count',
            color='sentiment',
            title="Sentiment Trend Over Time",
            color_discrete_map={
                'positive': '#2E8B57',
                'negative': '#DC143C',
                'neutral': '#FFD700'
            }
        )
    else:
        fig_line = go.Figure()
        fig_line.add_annotation(text="No year data available", x=0.5, y=0.5, showarrow=False)
    
    return fig_pie, fig_bar, fig_line

def create_wordcloud(text_data, title="Word Cloud"):
    """Create word cloud from text data"""
    if not text_data or len(text_data) == 0:
        return None
    
    # Combine all text
    combined_text = " ".join([str(text) for text in text_data if text and not pd.isna(text) and str(text).strip()])
    
    if not combined_text.strip():
        return None
    
    try:
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis',
            max_words=100
        ).generate(combined_text)
        
        import matplotlib.pyplot as plt
        from io import BytesIO
        import base64
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title)
        
        # Convert to base64 for Plotly
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        # Create Plotly figure
        fig_plotly = go.Figure()
        fig_plotly.add_layout_image(
            dict(
                source=f"data:image/png;base64,{img_str}",
                xref="paper", yref="paper",
                x=0, y=1, sizex=1, sizey=1,
                sizing="stretch", opacity=1, layer="below"
            )
        )
        fig_plotly.update_layout(
            title=title,
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            width=800, height=400
        )
        return fig_plotly
        
    except Exception as e:
        # Fallback to simple text display
        st.write(f"Word cloud generation failed: {e}")
        if len(combined_text) > 100:
            # Show most common words as text
            words = combined_text.lower().split()
            word_freq = {}
            for word in words:
                if len(word) > 3:  # Skip short words
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
            st.write("Most common words:", ", ".join([f"{word} ({count})" for word, count in top_words]))
        
        return None

def main():
    st.set_page_config(
        page_title="Employee Review Analytics Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("🏢 Employee Review Analytics Dashboard")
    st.sidebar.title("🔍 Filters & Controls")
    
    # Load data
    if 'df' not in st.session_state:
        with st.spinner("Loading and processing review data..."):
            st.session_state.df = load_and_process_data()
    
    df = st.session_state.df
    
    if df.empty:
        st.error("No data available. Please check if merged_reviews.json exists.")
        return
    
    # Sidebar filters
    st.sidebar.header("📋 Filter Reviews")
    
    # User type filter
    user_types = ['All'] + list(df['user_type'].unique())
    selected_user_type = st.sidebar.selectbox("User Type", user_types)
    
    # Sentiment filter
    sentiments = ['All'] + list(df['sentiment'].unique())
    selected_sentiment = st.sidebar.selectbox("Sentiment", sentiments)
    
    # Year filter
    years = df['year'].dropna().unique()
    if len(years) > 0:
        years = sorted([int(y) for y in years if not pd.isna(y)])
        year_range = st.sidebar.slider(
            "Year Range",
            min_value=min(years),
            max_value=max(years),
            value=(min(years), max(years))
        )
    else:
        year_range = None
        st.sidebar.info("No year data available for filtering")
    
    # Source filter
    sources = ['All'] + list(df['source'].unique())
    selected_source = st.sidebar.selectbox("Review Source", sources)
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_user_type != 'All':
        filtered_df = filtered_df[filtered_df['user_type'] == selected_user_type]
    
    if selected_sentiment != 'All':
        filtered_df = filtered_df[filtered_df['sentiment'] == selected_sentiment]
    
    if year_range and 'year' in filtered_df.columns and not filtered_df['year'].isna().all():
        filtered_df = filtered_df[
            (filtered_df['year'] >= year_range[0]) & 
            (filtered_df['year'] <= year_range[1])
        ]
    
    if selected_source != 'All':
        filtered_df = filtered_df[filtered_df['source'] == selected_source]
    
    # Ensure we have data after filtering
    if len(filtered_df) == 0:
        st.warning("No reviews match the selected filters. Please adjust your filter criteria.")
        return
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Reviews", len(filtered_df))
    
    with col2:
        positive_pct = len(filtered_df[filtered_df['sentiment'] == 'positive']) / len(filtered_df) * 100 if len(filtered_df) > 0 else 0
        st.metric("Positive Reviews", f"{positive_pct:.1f}%")
    
    with col3:
        avg_rating = filtered_df['overall_rating'].mean() if len(filtered_df) > 0 else 0
        st.metric("Average Rating", f"{avg_rating:.1f}/5")
    
    with col4:
        negative_count = len(filtered_df[filtered_df['sentiment'] == 'negative'])
        st.metric("Negative Reviews", negative_count)
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "🔍 Topic Analysis", 
        "☁️ Word Clouds", 
        "🔎 Review Search", 
        "💡 Recommendations"
    ])
    
    with tab1:
        st.header("📈 Sentiment Analysis Overview")
        
        if len(filtered_df) > 0:
            # Create sentiment charts
            fig_pie, fig_bar, fig_line = create_sentiment_charts(filtered_df)
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_pie, use_container_width=True)
            with col2:
                st.plotly_chart(fig_bar, use_container_width=True)
            
            st.plotly_chart(fig_line, use_container_width=True)
            
            # Rating distribution
            st.subheader("⭐ Rating Distribution")
            rating_dist = filtered_df['overall_rating'].value_counts().sort_index()
            fig_rating = px.bar(
                x=rating_dist.index,
                y=rating_dist.values,
                title="Distribution of Overall Ratings",
                labels={'x': 'Rating', 'y': 'Count'}
            )
            st.plotly_chart(fig_rating, use_container_width=True)
        else:
            st.warning("No data available for the selected filters.")
    
    with tab2:
        st.header("🎯 Topic Modeling Analysis")
        
        if len(filtered_df) > 0:
            analyzer = ReviewAnalyzer()
            
            # Topic modeling by sentiment
            for sentiment in ['positive', 'negative', 'neutral']:
                sentiment_reviews = filtered_df[filtered_df['sentiment'] == sentiment]
                if len(sentiment_reviews) > 2:  # Need at least 3 reviews for meaningful topics
                    st.subheader(f"📝 {sentiment.title()} Review Topics")
                    
                    texts = sentiment_reviews['text_content'].dropna().tolist()
                    # Filter out empty texts
                    texts = [text for text in texts if str(text).strip()]
                    
                    if len(texts) >= 2:
                        topics, topic_weights = analyzer.perform_topic_modeling(texts, n_topics=3)
                        
                        if topics:
                            for i, topic in enumerate(topics):
                                st.write(f"**{topic}**")
                        else:
                            st.write("Unable to extract meaningful topics from this data")
                    else:
                        st.write("Insufficient text data for topic modeling")
                elif len(sentiment_reviews) > 0:
                    st.write(f"Too few {sentiment} reviews for topic modeling (need at least 3)")
        else:
            st.warning("No data available for topic analysis.")
    
    with tab3:
        st.header("☁️ Word Cloud Analysis")
        
        if len(filtered_df) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("😊 Positive Reviews")
                positive_reviews = filtered_df[filtered_df['sentiment'] == 'positive']
                if len(positive_reviews) > 0:
                    pos_wordcloud = create_wordcloud(
                        positive_reviews['text_content'].tolist(),
                        "Positive Reviews Word Cloud"
                    )
                    if pos_wordcloud:
                        st.plotly_chart(pos_wordcloud, use_container_width=True)
                    else:
                        # Fallback: show most common words as text
                        all_text = " ".join(positive_reviews['text_content'].astype(str))
                        if all_text.strip():
                            words = all_text.lower().split()
                            word_freq = {}
                            for word in words:
                                if len(word) > 3:
                                    word_freq[word] = word_freq.get(word, 0) + 1
                            top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:15]
                            st.write("**Most common words in positive reviews:**")
                            st.write(", ".join([f"{word} ({count})" for word, count in top_words]))
                else:
                    st.write("No positive reviews found.")
            
            with col2:
                st.subheader("😞 Negative Reviews")
                negative_reviews = filtered_df[filtered_df['sentiment'] == 'negative']
                if len(negative_reviews) > 0:
                    neg_wordcloud = create_wordcloud(
                        negative_reviews['text_content'].tolist(),
                        "Negative Reviews Word Cloud"
                    )
                    if neg_wordcloud:
                        st.plotly_chart(neg_wordcloud, use_container_width=True)
                    else:
                        # Fallback: show most common words as text
                        all_text = " ".join(negative_reviews['text_content'].astype(str))
                        if all_text.strip():
                            words = all_text.lower().split()
                            word_freq = {}
                            for word in words:
                                if len(word) > 3:
                                    word_freq[word] = word_freq.get(word, 0) + 1
                            top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:15]
                            st.write("**Most common words in negative reviews:**")
                            st.write(", ".join([f"{word} ({count})" for word, count in top_words]))
                else:
                    st.write("No negative reviews found.")
        else:
            st.warning("No data available for word cloud generation.")
    
    with tab4:
        st.header("🔍 Review Search & Browse")
        
        # Search functionality
        search_query = st.text_input("🔎 Search reviews (keywords, job titles, etc.)")
        
        if search_query:
            search_results = filtered_df[
                filtered_df['text_content'].str.contains(search_query, case=False, na=False) |
                filtered_df['title'].str.contains(search_query, case=False, na=False) |
                filtered_df['job_title'].str.contains(search_query, case=False, na=False)
            ]
        else:
            search_results = filtered_df
        
        # Display results
        st.write(f"Found {len(search_results)} reviews")
        
        # Sort options
        sort_by = st.selectbox("Sort by", ["Recent", "Rating (High to Low)", "Rating (Low to High)", "Helpful"])
        
        if sort_by == "Recent":
            search_results = search_results.sort_values('submission_date', ascending=False)
        elif sort_by == "Rating (High to Low)":
            search_results = search_results.sort_values('overall_rating', ascending=False)
        elif sort_by == "Rating (Low to High)":
            search_results = search_results.sort_values('overall_rating', ascending=True)
        elif sort_by == "Helpful":
            search_results = search_results.sort_values('helpful_count', ascending=False)
        
        # Display reviews
        for idx, row in search_results.head(10).iterrows():
            with st.expander(f"⭐ {row['overall_rating']}/5 - {row['title']} ({row['sentiment']})"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**Job Title:** {row['job_title']}")
                    st.write(f"**Location:** {row['location']}")
                    st.write(f"**User Type:** {row['user_type']}")
                    st.write(f"**Source:** {row['source']}")
                    st.write("**Review:**")
                    st.write(row['text_content'][:500] + "..." if len(row['text_content']) > 500 else row['text_content'])
                
                with col2:
                    st.write(f"**Date:** {row['submission_date']}")
                    st.write(f"**Helpful:** {row['helpful_count']}")
                    sentiment_color = {'positive': '🟢', 'negative': '🔴', 'neutral': '🟡'}
                    st.write(f"**Sentiment:** {sentiment_color.get(row['sentiment'], '⚪')} {row['sentiment']}")
    
    with tab5:
        st.header("💡 AI-Powered Recommendations")
        
        negative_reviews = filtered_df[filtered_df['sentiment'] == 'negative']
        
        if len(negative_reviews) > 0:
            st.subheader("🔍 Critical Issues Analysis")
            
            # Show negative review statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Negative Reviews", len(negative_reviews))
                
                # Most common issues in negative reviews
                st.subheader("🚨 Common Issues")
                issue_keywords = ['management', 'salary', 'work-life', 'toxic', 'communication', 'career', 'benefits']
                issue_counts = {}
                
                for keyword in issue_keywords:
                    count = negative_reviews['text_content'].str.contains(keyword, case=False, na=False).sum()
                    if count > 0:
                        issue_counts[keyword] = count
                
                if issue_counts:
                    for issue, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True):
                        st.write(f"• **{issue.title()}**: {count} mentions")
            
            with col2:
                # User types with most negative reviews
                negative_by_type = negative_reviews['user_type'].value_counts()
                fig_negative = px.bar(
                    x=negative_by_type.values,
                    y=negative_by_type.index,
                    orientation='h',
                    title="Negative Reviews by User Type"
                )
                st.plotly_chart(fig_negative, use_container_width=True)
            
            # Generate AI recommendations
            st.subheader("🤖 AI-Generated Improvement Recommendations")
            
            if st.button("Generate Recommendations", type="primary"):
                with st.spinner("Analyzing negative reviews and generating recommendations..."):
                    # Prepare data for AI analysis
                    sample_negative_reviews = negative_reviews.head(10)['text_content'].tolist()
                    
                    prompt = f"""
                    As an HR and organizational development expert, analyze these employee reviews and provide specific, actionable recommendations for improvement:

                    NEGATIVE REVIEW SAMPLES:
                    {chr(10).join([f"- {review[:200]}..." for review in sample_negative_reviews])}

                    Please provide:
                    1. TOP 3 CRITICAL ISSUES identified from these reviews
                    2. SPECIFIC ACTION ITEMS for each issue (be concrete and measurable)
                    3. PRIORITY LEVEL for each recommendation (High/Medium/Low)
                    4. TIMELINE for implementation
                    5. SUCCESS METRICS to track improvement

                    Format your response clearly with headers and bullet points.
                    """
                    
                    recommendations = call_gemini_api(prompt)
                    st.markdown(recommendations)
            
            # Export functionality
            st.subheader("📥 Export Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Export Negative Reviews CSV"):
                    csv = negative_reviews.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"negative_reviews_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("Export Summary Report"):
                    summary = f"""
                    EMPLOYEE REVIEW ANALYSIS SUMMARY
                    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
                    
                    OVERVIEW:
                    - Total Reviews Analyzed: {len(filtered_df)}
                    - Negative Reviews: {len(negative_reviews)} ({len(negative_reviews)/len(filtered_df)*100:.1f}%)
                    - Average Rating: {filtered_df['overall_rating'].mean():.2f}/5
                    
                    TOP ISSUES:
                    {chr(10).join([f"- {issue}: {count} mentions" for issue, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]])}
                    
                    MOST AFFECTED USER TYPES:
                    {chr(10).join([f"- {user_type}: {count} negative reviews" for user_type, count in negative_by_type.head(3).items()])}
                    """
                    
                    st.download_button(
                        label="Download Summary",
                        data=summary,
                        file_name=f"review_summary_{datetime.now().strftime('%Y%m%d')}.txt",
                        mime="text/plain"
                    )
        else:
            st.info("No negative reviews found with current filters.")
    
    # Sidebar: Keyword tracking
    st.sidebar.header("🎯 Keyword Tracking")
    
    keyword_to_track = st.sidebar.text_input("Track keyword alerts")
    if keyword_to_track:
        keyword_mentions = filtered_df[
            filtered_df['text_content'].str.contains(keyword_to_track, case=False, na=False)
        ]
        
        if len(keyword_mentions) > 0:
            st.sidebar.success(f"'{keyword_to_track}' found in {len(keyword_mentions)} reviews")
            
            # Show sentiment breakdown for keyword
            keyword_sentiment = keyword_mentions['sentiment'].value_counts()
            for sentiment, count in keyword_sentiment.items():
                color = {'positive': '🟢', 'negative': '🔴', 'neutral': '🟡'}
                st.sidebar.write(f"{color.get(sentiment, '⚪')} {sentiment}: {count}")
        else:
            st.sidebar.info(f"No mentions of '{keyword_to_track}' found")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("📊 Dashboard powered by Streamlit & AI")

if __name__ == "__main__":
    main()
