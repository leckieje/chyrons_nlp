# sklearn pipeline 
import pandas as pd 
import numpy as np
import random
import string
import re
import nltk
from nltk.corpus import stopwords, words
from nltk.corpus import words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# nltk.download('words')

'''
---> To get count and tfidf vectors for summer 2020/MSNBC/Fox:

chy_summer = load_summer()
chy_nets = get_network_pair(chy_summer, net_one='FOXNEWSW', net_two='MSNBCW')
feature_count, count_vec = clean_counts(chy_nets, n_grams=(1,1))
feature_tfidf, tfidf_vec = clean_tfidf(chy_nets, n_grams=(1,1))

OR:

chy_summer = load_summer()

chy_nets = get_network_pair(chy_summer, net_one='FOXNEWSW', net_two='MSNBCW')
feature_count, count_vec = clean_counts(chy_nets, n_grams=(1,2))

fox_features, fox_counts = clean_counts(chy_nets.loc[chy_nets['channel'] == 'FOXNEWSW'], n_grams=(1,2))
msnbc_features, msnbc_counts = clean_counts(chy_nets.loc[chy_nets['channel'] == 'MSNBCW'], n_grams=(1,2))
fox_df = get_dataframe(fox_counts, fox_features)
msnbc_df = get_dataframe(msnbc_counts, msnbc_features)
'''

# sklearn pipeline 

# english dictionary 
# eng_dict = words.words()

# get a random sample of chyrons
def get_random_sample(df, num_samples):
    idx_lst = [num for num in range(len(df))]
    idx_samps = random.sample(idx_lst, num_samples)
    df_samp = df.iloc[idx_samps]
    return df_samp

# get stop words
custom_stops = ['msnbc', 'nbc', 'nbc news', 'mssnbc news', 'fox', 'fox news', 
                'says', 'er', 'tn', 'ui', 'thf', 'thf', 'nm', 'ni', 'new', 'nl'
                'li', 'iu', 'ai', 'ul', 'wsntunnelztowersdrg', 'yingst', 'wnr'
                'irl', 'wopi', 'across', 'xeljanz']

def get_stop_words(words, custom_stops=custom_stops):
    sw = stopwords.words('english')
    stops = words[0]
    
    for lst in words[1:]:
        stops += lst

    stops = stops + custom_stops
    
    return sw + stops

## ------------------------------------------------------ ##

# sklearn count vectorizer
def get_countvec(corpus, stop_words='english', min_df=.01, n_grams=(1,1)):
    vectorizer = CountVectorizer(stop_words=stop_words, min_df=min_df, ngram_range=n_grams)
    X = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names()
    
    return feature_names, X.toarray()

# sklearn tfidf vectorizer
def get_tfidf(corpus, max_features=None, min_df=.01, stop_words='english', n_grams=(1,1)):
    vectorizer = TfidfVectorizer(max_features=None, min_df=min_df, max_df=1.0, stop_words='english', ngram_range=n_grams)
    X = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names()
    
    return feature_names, X.toarray()

# vectorizer to dataframe
def get_dataframe(X, feature_names):
    df = pd.DataFrame(data = X, columns = feature_names)
    return df

## ------------------------------------------------------ ##

# odd "word" functions
def get_numeric_words(feature_names):
    stops = set()
    nums = [str(num) for num in range(10)]
    
    for word in feature_names:
        for char in word:
            if char in nums:
                stops.add(word)
                break
                
    return list(stops)

def get_non_alpha_start(feature_names):
    non_alpha = []
    alpha = list(string.ascii_lowercase)
    for name in feature_names:
        if name[0] not in alpha:
            non_alpha.append(name)
            
    return non_alpha

def get_underscores(feature_names):
    unders = []
    for name in feature_names:
        if '_' in name:
            unders.append(name)
            
    return unders

def get_multis(feature_names):
    multi = []
    for name in feature_names:
        if len(name) == 2:
            if name[0] == name[1]:
                multi.append(name)
        else:
            search = re.search(r"([A-Za-z])\1\1", name)
            if search != None:
                multi.append(name)
                
    return multi

# wrap for above 
def get_special_stops(feature_names):
    numeric = get_numeric_words(feature_names)
    non_alpha = get_non_alpha_start(feature_names)
    unders = get_underscores(feature_names)
    multi_lett = get_multis(feature_names)
    
    return [numeric, non_alpha, unders, multi_lett]


## ------------------------------------------------------ ##

# PIPELINE:

def load_samps(num_samples):
    chyrons = pd.read_csv('/Users/jonleckie/Desktop/DSI_all/capstones/capstone_two/chyrons/chyron_all.csv') # read all data
    chyrons.drop(472372, inplace=True) # drop single NaN value
    chy_samp = get_random_sample(chyrons, num_samples) # random sample for working locally 
    
    return chy_samp
    
def clean_counts(chy_samp, min_df=.001, n_grams=(1,1)):
    count_features, chyron_counts = get_countvec(chy_samp['text'], stop_words='english', min_df=min_df, n_grams=n_grams) # get primary feature names
    stops = get_special_stops(count_features) # get additional stop words
    stop_words_plus = get_stop_words(stops) # add to nltk stop words
    count_features, chyron_counts = get_countvec(chy_samp['text'], stop_words=stop_words_plus, min_df=min_df, n_grams=n_grams) # get new vector matrix

    return count_features, chyron_counts

def clean_tfidf(chy_samp, min_df=.001, n_grams=(1,1)):
    tfidf_features, chyron_tfidf = get_tfidf(chy_samp['text'], max_features=None, min_df=min_df, stop_words='english', n_grams=n_grams) # get primary feature names
    stops = get_special_stops(tfidf_features) # get additional stop words
    stop_words_plus = get_stop_words(stops) # add to nltk stop words
    tfidf_features, chyron_tfidf = get_tfidf(chy_samp['text'], max_features=None, min_df=min_df, stop_words=stop_words_plus, n_grams=n_grams) # get new vector matrix

    return tfidf_features, chyron_tfidf


# limit data from summer 2020, MSNBC and Fox
def load_summer(start_date='2020-05-25', end_date='2020-11-03'):
    chyrons = pd.read_csv('chyron_all.csv')
    chyrons.drop(472372, inplace=True)
    chyrons = chyrons.replace(r'\\n',' ',regex=True)
    chyrons = chyrons.replace(r'\\u',' ',regex=True)
    chy_summer = chyrons.loc[(chyrons['date_time_(UTC)'] >= start_date) & (chyrons['date_time_(UTC)'] <= end_date)]
    
    return chy_summer

def get_network_pair(df, net_one='FOXNEWSW', net_two='MSNBCW'):
    chy_nets = df.loc[(df['channel'] == net_one) | (df['channel'] == net_two)]
    
    return chy_nets

## ------------------------------------------------------ ##

# for word frequency charting
def get_word_freq(df):
    series = df.sum()
    series = series.apply(lambda x: x/len(series))
    return series

def get_freq_df(fox, msnbc):
#     words = [word for words in fox.index] + [word for word in msnbc.index]
    shared_words = []
    freq = []
    diff = []
    
    for word in fox.index:
        if word in msnbc.index:
            shared_words.append(word)
            freq.append((fox[word], msnbc[word]))  
            diff.append(np.abs(fox[word] - msnbc[word]))
            
    freq_df = pd.DataFrame({'word': shared_words, 'fox_freq':[freq[0] for freq in freq],
                           'msnbc_freq': [freq[1] for freq in freq], 'difference': diff})
            
    return freq_df

def freq_df(fox_df, msnbc_df):
    fox_freq = get_word_freq(fox_df)
    msnbc_freq = get_word_freq(msnbc_df)
    df = get_freq_df(fox_freq, msnbc_freq)
    df.sort_values(by='difference', ascending=False, axis=0, inplace=True)
    
    return df

# chart consecutive word frequencies
def chart_freq_diff(df):
    fig, ax = plt.subplots()
    labels = df['word']
    fox = df['fox_freq']
    msnbc = df['msnbc_freq']
    
    x = np.arange(len(labels))
    width = 0.35
    fox_bar = ax.bar(x - width/2, fox, width, label='Fox News')
    msnbc_bar = ax.bar(x + width/2, msnbc, width, label='MSNBC')
    
    ax.set_ylabel('Frequency')
    ax.set_title('Word Use Frequency')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout();

# chart by choice of words
def chart_word_diff(df, word_lst):
    fig, ax = plt.subplots()
    labels = word_lst
    df_ = df.set_index('word')
    fox = df_.loc[word_lst, 'fox_freq']
    msnbc = df_.loc[word_lst, 'msnbc_freq']
    
    x = np.arange(len(labels))
    width = 0.35
    fox_bar = ax.bar(x - width/2, fox, width, label='Fox News')
    msnbc_bar = ax.bar(x + width/2, msnbc, width, label='MSNBC')
    
    ax.set_ylabel('Frequency')
    ax.set_title('Word Use Frequency')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout();