# 📊 Employee Review Analytics Dashboard

An interactive dashboard for analyzing employee reviews with sentiment analysis, topic modeling, and AI-powered recommendations.

## 🌟 Features

### 📈 Core Analytics
- **Sentiment Classification**: Automatically classifies reviews as positive, negative, or neutral
- **User Type Classification**: Categorizes reviewers (Intern, Senior/Manager, Consultant, etc.)
- **Interactive Filtering**: Filter by user type, sentiment, year, and review source
- **Real-time Metrics**: Overview of key statistics and trends

### 🎯 Advanced Analysis
- **Topic Modeling**: Identifies key themes in reviews using LDA (Latent Dirichlet Allocation)
- **Word Clouds**: Visual representation of most common words in positive/negative reviews
- **Trend Analysis**: Sentiment trends over time
- **Rating Distribution**: Detailed breakdown of review ratings

### 🔍 Search & Browse
- **Smart Search**: Search reviews by keywords, job titles, or content
- **Advanced Sorting**: Sort by recency, rating, helpfulness
- **Review Details**: Expandable view with full review content and metadata

### 💡 AI-Powered Recommendations
- **Critical Issues Analysis**: Identifies most common problems from negative reviews
- **Gemini AI Integration**: Generates specific, actionable improvement recommendations
- **Export Capabilities**: Download filtered data and summary reports
- **Keyword Tracking**: Monitor specific terms and get alerts

### 📊 Visualizations
- **Interactive Charts**: Built with Plotly for responsive, interactive visualizations
- **Sentiment Distribution**: Pie charts, bar charts, and trend lines
- **User Type Analysis**: Breakdown by employee categories
- **Rating Analytics**: Comprehensive rating distribution analysis

## 🚀 Quick Start

### Option 1: Windows Batch File (Easiest)
1. Double-click `run_dashboard.bat`
2. The script will automatically install dependencies and start the dashboard
3. Open your browser to `http://localhost:8501`

### Option 2: Python Setup Script
```bash
python setup_dashboard.py
```

### Option 3: Manual Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run dashboard.py
```

## 📋 Requirements

### System Requirements
- Python 3.8 or higher
- Windows, macOS, or Linux
- Internet connection (for AI recommendations)

### Data Requirements
- `merged_reviews.json` file in the same directory
- The file should contain the merged review data from Glassdoor and Indeed

### Python Dependencies
```
streamlit>=1.28.0
pandas>=1.5.0
plotly>=5.15.0
numpy>=1.21.0
textblob>=0.17.1
wordcloud>=1.9.2
scikit-learn>=1.3.0
nltk>=3.8
requests>=2.28.0
python-dateutil>=2.8.2
```

## 🎛️ Dashboard Usage

### 🔍 Filters Panel (Left Sidebar)
- **User Type**: Filter by employee category (Intern, Senior/Manager, etc.)
- **Sentiment**: Show only positive, negative, or neutral reviews
- **Year Range**: Select specific time periods
- **Review Source**: Filter by Glassdoor or Indeed
- **Keyword Tracking**: Monitor specific terms with real-time alerts

### 📊 Main Dashboard Tabs

#### 1. Overview Tab
- Overall sentiment distribution
- Sentiment by user type
- Sentiment trends over time
- Rating distribution analysis

#### 2. Topic Analysis Tab
- Automated topic modeling for each sentiment category
- Key themes and patterns identification
- Topic distribution visualization

#### 3. Word Clouds Tab
- Visual representation of most frequent words
- Separate clouds for positive and negative reviews
- Interactive and responsive design

#### 4. Review Search Tab
- Powerful search functionality
- Sort and filter capabilities
- Detailed review browsing
- Expandable review cards with full metadata

#### 5. Recommendations Tab
- AI-powered analysis of negative reviews
- Critical issues identification
- Actionable improvement suggestions
- Export and reporting capabilities

## 🤖 AI Integration

### Gemini API Features
- **Smart Analysis**: Uses Google's Gemini AI for intelligent review analysis
- **Actionable Insights**: Generates specific, implementable recommendations
- **Priority Assessment**: Categorizes issues by importance and urgency
- **Success Metrics**: Suggests measurable outcomes for tracking improvement

### API Configuration
The dashboard uses the provided Gemini API key: `AIzaSyDmPiDZGWkap0oKZzENszs9nsK-E-A2qMA`

## 📁 File Structure

```
Sentiment Analysis/
├── dashboard.py              # Main dashboard application
├── setup_dashboard.py       # Python setup script
├── run_dashboard.bat        # Windows launcher
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── merge_reviews.py        # Review merging script
├── merged_reviews.json     # Processed review data
├── Glassdoor.json         # Original Glassdoor data
└── Indeed.json            # Original Indeed data
```

## 🔧 Technical Details

### Data Processing Pipeline
1. **Data Loading**: Reads merged JSON review data
2. **Text Processing**: Cleans and preprocesses review text
3. **Sentiment Analysis**: Uses VADER and TextBlob for sentiment scoring
4. **Classification**: Categorizes users based on job titles and employment data
5. **Topic Modeling**: Applies LDA for theme extraction
6. **Visualization**: Generates interactive charts and graphs

### Machine Learning Components
- **Sentiment Analysis**: Hybrid approach using VADER + TextBlob
- **Topic Modeling**: Latent Dirichlet Allocation (LDA)
- **Text Vectorization**: TF-IDF for feature extraction
- **Clustering**: K-means for review grouping

### Performance Optimization
- **Caching**: Streamlit session state for data persistence
- **Lazy Loading**: Components loaded on demand
- **Efficient Processing**: Vectorized operations with pandas/numpy
- **Memory Management**: Optimized data structures

## 🚨 Troubleshooting

### Common Issues

#### "No module named 'streamlit'"
```bash
pip install streamlit
```

#### "merged_reviews.json not found"
- Ensure the data file is in the same directory as the dashboard
- Run the merge script first if needed

#### Dashboard won't start
- Check Python version (3.8+ required)
- Verify all dependencies are installed
- Check firewall settings for port 8501

#### AI recommendations not working
- Verify internet connection
- Check if API key is valid
- Review API quota limits

### Performance Tips
- Use filters to reduce dataset size for faster processing
- Close other browser tabs when running the dashboard
- Restart the dashboard if it becomes slow

## 📈 Analytics Capabilities

### Sentiment Metrics
- **Overall Distribution**: Percentage breakdown of sentiment categories
- **Temporal Trends**: How sentiment changes over time
- **User Type Analysis**: Sentiment patterns by employee category
- **Source Comparison**: Differences between Glassdoor and Indeed reviews

### Topic Analysis
- **Automatic Theme Detection**: Identifies key discussion topics
- **Sentiment-Specific Topics**: Different themes for positive/negative reviews
- **Word Frequency Analysis**: Most common terms and phrases
- **Trend Identification**: Emerging issues and positive developments

### Recommendation Engine
- **Issue Prioritization**: Ranks problems by frequency and impact
- **Action Planning**: Specific steps for improvement
- **Success Tracking**: Metrics for measuring progress
- **Timeline Planning**: Realistic implementation schedules

## 🎯 Use Cases

### HR Teams
- Monitor employee satisfaction trends
- Identify systemic issues early
- Generate improvement action plans
- Track progress over time

### Management
- Understand team sentiment
- Prioritize workplace improvements
- Benchmark against industry standards
- Make data-driven decisions

### Executives
- High-level satisfaction overview
- Strategic planning insights
- Risk identification
- Culture assessment

## 🔒 Privacy & Security

- All processing is done locally
- No sensitive data is transmitted (except to Gemini for recommendations)
- Review anonymization built-in
- Secure API key handling

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve the dashboard.

## 📄 License

This project is for internal use and analysis purposes.

---

**Happy Analyzing! 📊✨**
