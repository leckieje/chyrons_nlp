from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix, roc_curve
from sklearn.decomposition import NMF
from sklearn.inspection import permutation_importance, plot_partial_dependence, partial_dependence

'''
---> To prep for models:

X_counts = get_dataframe(count_vec, feature_count) # --> Count X
X_tfidf = get_dataframe(tfidf_vec, feature_tfidf) # --> Tfidf X
y = chy_nets['channel'] 

# first split for train and holdout sets:
X_train, X_test, y_train, y_test = train_test_split(X_counts, y, 
                                                    train_size=0.33, 
                                                    shuffle=True, stratify=y)
X_train_T, X_test_T, y_train_T, y_test_T = train_test_split(X_tfidf, y, 
                                                            train_size=0.33, 
                                                            shuffle=True, stratify=y)

'''

# Naive Bayes


# ROC Curve
def plot_roc(X, y, vec_type='Count'):
    y_net = [1 if net == 'MSNBCW' else 0 for net in y]
    X_train, X_test, y_train, y_test = train_test_split(X, y_net, train_size=0.33, shuffle=True, stratify=y)
    fig, ax = plt.subplots()
    
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_hat = model.predict_proba(X_test)
    fpr, tpr, thresh = roc_curve(y_test, y_hat[:, 1])
    
    ax.plot(fpr, tpr)
    ax.plot([0,1], [0,1], ls='--', color='k')
    ax.set_title(f'Naive Bayes ROC Curve {vec_type}')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    
    return thresh

# Random Forest


# ROC Curve
def plot_roc_forest(X, y, vec_type='(Count)', max_depth=10):
    y_net = [1 if net == 'MSNBCW' else 0 for net in y]
    X_train, X_test, y_train, y_test = train_test_split(X, y_net, train_size=0.33, shuffle=True, stratify=y)
    fig, ax = plt.subplots()
    
    model = RandomForestClassifier(max_depth=max_depth, max_features='log2', n_jobs=-1)
    model.fit(X_train, y_train)
    y_hat = model.predict_proba(X_test)
    fpr, tpr, thresh = roc_curve(y_test, y_hat[:,1])
    
    ax.plot(fpr, tpr)
    ax.plot([0,1], [0,1], ls='--', color='k')
    ax.set_title(f'Random Forest ROC Curve {vec_type}')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    
    return thresh

# feature importances
# gini
def chart_gini_import(model, X, size=15):
    fig, ax = plt.subplots()
    feature_scores = pd.Series(model.feature_importances_, index=X.columns)
    feature_scores = feature_scores[:num].sort_values()
    ax = feature_scores.plot(kind='barh', figsize=(10,4))
    ax.set_title('Gini Importance')
    ax.set_xlabel('Avg. Contribution to Info Gain');

# permutations
def chart_permutation_import(model, X, y, size=15, n_repeats=5):
    fig, ax = plt.subplots()
    perms = permutation_importance(model, X, y, n_repeats=n_repeats)
    sorted_idx = perms.importances_mean.argsort()
    ax.boxplot(perms.importances[sorted_idx[:size]].T, vert=False, labels=X.columns[sorted_idx])
    ax.set_title('Permutation Importance');


# other charting functions:
# top words
def chart_top(word_lst, network, low=0, high=10,):
    fig, ax = plt.subplots(figsize=(10,4))
    
    ax.barh(word_lst.index[low:high], word_lst.values[low:high], align='center')
    ax.invert_yaxis()
    ax.set_title(f'Top Word Counts: {network}')
    ax.set_ylabel('Words')
    ax.set_xlabel('Counts');

# shared words (network word counts against each other)
def chart_shared_word_counts(thresh=20000):
    fig, ax = plt.subplots()
    nbc_lt = []
    fox_lt = []
    words = []
    
    for i, word in enumerate(nbc_word_counts):
        if word < thresh:
            nbc_lt.append(nbc_word_counts[i])
            fox_lt.append(fox_word_counts[i])
        
    ax.scatter(nbc_lt, fox_lt, alpha=0.3)
    ax.set_title('Shared Word Counts by Network')
    ax.set_xlabel('MSNBC')
    ax.set_ylabel('Fox News');