from pathlib import Path
import glob
import numpy as np
import pandas as pd
import collections
import num2words
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.graph_objects as go
import pickle
sid = SentimentIntensityAnalyzer()
#import plotly.graph_objects as px

import json
import plotly
import plotly.express as px
data = pd.read_pickle('resources/Data.pkl')
def get_date_time(K):
    fruit = [i for i in K]
    amount = [K[i] for i in K]
    df = pd.DataFrame({
        "Date": fruit,
        "Reviews": amount,

    })
    df = df.sort_values(by="Date")
    fig = px.line(df, x="Date", y="Reviews", title="Timeline of Reviews")

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    #header =
    return graphJSON

def pree(text):

    new_data = data[data['Phone_names']==text]
    list = len(new_data['extract'])
    K = collections.Counter(new_data['Countries'])
    L = collections.Counter(new_data['lang'])
    D = collections.Counter(new_data['date'])
    R = collections.Counter(new_data['Ratings'])
    R_C = get_sentiments(new_data)
    S = collections.Counter(new_data['source'])
    your_list, graph = aspect_gainer(text)
    return list,K,L,D,R,R_C,S,your_list,graph

def country_graph(K):
    fruit = [i for i in K]
    amount = [K[i] for i in K]
    df = pd.DataFrame({
        "Country": fruit,
        "Reviews": amount,

    })

    fig = px.bar(df, x ="Country", y="Reviews", title="Number of Reviews by country")

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    #header =
    return graphJSON


def generate_graph_language(K):
    fruit = [i for i in K]
    amount = [K[i] for i in K]
    df = pd.DataFrame({
        "Language": fruit,
        "Reviews": amount,

    })

    fig = px.bar(df, x="Language", y="Reviews", title='Number of Reviews by Language')

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    #header = "Number of Reviews by Language"
    return graphJSON



def rating_graph(K):
    fruit = [i for i in K]
    amount = [K[i] for i in K]
    df = pd.DataFrame({
        "Ratings": fruit,
        "Numbers": amount,

    })
    df = df.sort_values(by="Ratings")
    fig = px.pie(df, names="Ratings", values="Numbers" , title='Ratings Distribution')

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    #header = "Ratings Distribution"
    return graphJSON




def sentiment_graph(K):
    fruit = [i for i in K]
    amount = [K[i] for i in K]
    df = pd.DataFrame({
        "Sentiment": fruit,
        "Values": amount,

    })

    fig = px.bar(df, x="Sentiment", y="Values", title='Sentiment Analysis')

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    # header = "Number of Reviews by Language"
    return graphJSON

def sellers_graph(K):
    fruit = [i for i in K]
    amount = [K[i] for i in K]
    size = [i*10 for i in amount]
    df = pd.DataFrame({
        "Seller": fruit,
        "Numbers": amount,

    })

    fig = px.scatter(df, x="Seller", y="Numbers",color="Seller",size="Numbers", title='Sellers Distribution')

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    # header = "Number of Reviews by Language"
    return graphJSON




def get_sentiments(new_df):
    scores = new_df['extract'].apply(lambda review: sid.polarity_scores(str(review)))
    compound = scores.apply(lambda score_dict: score_dict['compound'])
    dic = {'Positive': 0, 'Negative': 0, 'Neutral': 0}

    for i in compound:
        if i < 0 or i == 0:
            dic['Negative'] += 1
        elif 0 < i < 0.5:
            dic['Neutral'] += 1
        else:
            dic['Positive'] += 1
    return dic

def aspect_gainer(text):
    open_file = open(f'Pickles/{text}.pkl', "rb")
    T = pickle.load(open_file)
    your_list = []
    graph = generate_graph(T)
    for i in range(len(T[0])):

        your_list.append((T[0][i],T[1][i]))


    return your_list,graph


def generate_graph(T):
    color = []
    x_bar = []
    y_bar = []
    N = T[0]
    M = T[2]
    for i in range(15):
        color.append('Negative')
        color.append('Positive')
        x_bar.append(N[i])
        x_bar.append(N[i])
        y_bar.append(M[i][0])
        y_bar.append(M[i][1])
    df = pd.DataFrame({
        'Aspect': x_bar,
        'Sentiment_numbers': y_bar,
        'sentiment': color

    })
    df = df.sample(frac=1).reset_index(drop=True)

    fig = px.bar(df, x='Aspect', y='Sentiment_numbers', color='sentiment', title='Aspect based Sentiments', )


    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graph_json