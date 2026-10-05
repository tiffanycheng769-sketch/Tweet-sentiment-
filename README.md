# Stock Market Crash 2022 – Tweet Sentiment Analysis

A Jupyter Notebook project that classify tweets during the 2022 stock market crash into **Positive**, **Neutral**, and **Negative** sentiments using PySpark and Spark NLP.

## Features

- Data loading and exploratory data analysis (EDA)
- Text preprocessing (URLs, mentions, hashtags, tickers, punctuation)
- Baseline model: TF-IDF + Logistic Regression
- Comparison with:
  - Random Forest
  - Naive Bayes
  - Decision Tree
- Word embedding models:
  - Word2Vec + Logistic Regression
  - GloVe + Logistic Regression
- Hyperparameter tuning with CrossValidator
- Ensemble models:
  - Voting Classifier
  - Stacking Classifier
- Visualizations:
  - Sentiment distribution
  - Word clouds (overall and by sentiment / POS)
  - Top hashtags (overall and by sentiment)
  - Top words and bigrams
  - Text length vs sentiment
  - Word frequency distribution

## Requirements

- Python 3.9+
- PySpark
- spark-nlp
- nltk
- wordcloud
- matplotlib
- pandas
- numpy

## Setup

1. Clone this repository:

   ```bash
   git clone https://github.com/YOUR_USERNAME/stock-market-sentiment.git
   cd stock-market-sentiment
   ```

2. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   # On macOS / Linux
   source venv/bin/activate
   # On Windows
   venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start a Jupyter server:

   ```bash
   jupyter notebook
   ```

5. Open `notebooks/sentiment_analysis.ipynb` and run all cells.

## Data

Dataset: `stock_market_crash_2022.csv`  
Source: https://www.kaggle.com/datasets/tejasurya/huge-stock-market-crash-2022 on Kaggle

Columns:

- `text`: tweet text
- `text_sentiment`: sentiment label  
  - `0` = Neutral  
  - `1` = Positive  
  - `2` = Negative


## Project Structure

```text
stock-market-sentiment/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ data/
│  └─ stock_market_crash_2022.csv  
└─ notebooks/
   └─ sentiment_analysis.ipynb
```

- `notebooks/sentiment_analysis.ipynb`: main analysis notebook (EDA, modeling, evaluation, visualizations)
- `data/`: directory for the dataset (ignored by Git)

## Usage

1. Put your `stock_market_crash_2022.csv` into the `data/` folder.
2. Open `notebooks/sentiment_analysis.ipynb` in Jupyter Notebook or JupyterLab.
3. Run all cells from top to bottom.

The notebook will:

- Load and preprocess the data
- Train and evaluate multiple models
- Perform hyperparameter tuning
- Build ensemble models
- Generate various visualizations

## Results

(You can fill this in once you have final numbers, for example:)

- Best single model: *Random Forest (TF-IDF)* with accuracy ≈ **54.65**
- Best ensemble model: *Voting Classifier* with accuracy ≈ **70.07%**
- Key observations:
  - richer word representations, which capture more context and meaning, can make a real difference compared to just counting word frequencies.
  - hard voting ensemble, which takes the majority prediction from the tuned Random Forest, Naïve Bayes and Decision Tree models, jumped to around 70% accuracy
  -  neutral comments tended to be shorter than the others
