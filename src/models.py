from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix, roc_curve
from sklearn.decomposition import NMF
from sklearn.inspection import permutation_importance, plot_partial_dependence, partial_dependence

import matplotlib.pyplot as plt
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
def eval_naive_bayes(X, y, folds=20, fit_prior=False):
    kf = KFold(n_splits=folds, shuffle=True)
    accuracy = []
    
    for train, test in kf.split(X):
        model = MultinomialNB(alpha=1, fit_prior=fit_prior)
        model.fit(X.iloc[train], y.iloc[train])
        accuracy.append(model.score(X.iloc[test], y.iloc[test]))
    
    return model, np.mean(accuracy)

def test_folds(X, y, fold_lst):
    accuracy = []
    iters = 0
    for folds in fold_lst:
        print(iters)
        model, score = eval_naive_bayes(X, y, folds=folds)
        accuracy.append(score)
        
        iters += 1
        
    return accuracy

def plot_naive_bayes(X, y, fold_lst):
    fig, ax = plt.subplots()
    xs = fold_lst
    ys = test_folds(X, y, fold_lst)
    ax.plot(xs, ys)
    ax.set_title('Naive Bayes Accuracy by Num_Folds')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Folds');


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
def eval_random_forest(X, y, folds=10, n_estimators=100, max_depth=5, max_leaf=None, max_features='log2'):
    kf = KFold(n_splits=folds, shuffle=True)
    accuracy = []
    oob = []
    iters = 0
    
    for train, test in kf.split(X):
        forest = RandomForestClassifier(n_estimators=n_estimators, 
                                        max_depth=max_depth, n_jobs=-1, 
                                        max_leaf_nodes=max_leaf, max_features=max_features, 
                                        oob_score=True)
        forest.fit(X.iloc[train], y.iloc[train])
        accuracy.append(forest.score(X.iloc[test], y.iloc[test]))
        oob.append(forest.oob_score_)
        
        print(iters)
        iters += 1
    
    return np.mean(accuracy), np.mean(oob), forest

def test_forest_folds(X, y, fold_lst):
    accuracy = []
    for folds in fold_lst:
        accuracy.append(eval_random_forest(X, y, folds=folds))
        
    return accuracy

def test_forest_depth(X, y, depth_lst):
    accuracy = []
    for depth in depth_lst:
        accuracy.append(eval_random_forest(X, y, max_depth=depth))
        
    return accuracy

def test_forest_estimators(X, y, est_lst):
    accuracy = []
    for est in est_lst:
        accuracy.append(eval_random_forest(X, y, n_estimators=est, max_depth=50))
        
    return accuracy

def test_max_leafs(X, y, leaf_lst):
    accuracy = []
    for leaf in leaf_lst:
        accuracy.append(eval_random_forest(X, y, max_leaf=leaf))
        
    return accuracy

def plot_folds_random_forest_folds(X, y, fold_lst):
    fig, ax = plt.subplots()
    xs = fold_lst
    ys = test_forest_folds(X, y, fold_lst)
    ax.plot(xs, ys)
    ax.set_title('Random Forest Accuracy by Folds')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Folds');
    
def plot_depth_random_forest(X, y, depth_lst):
    fig, ax = plt.subplots()
    xs = depth_lst
    ys = test_forest_depth(X, y, depth_lst)
    ax.plot(xs, ys)
    ax.set_title('Random Forest Accuracy by Depth')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Depth');

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
    feature_scores = feature_scores.sort_values()
    ax = feature_scores[:num].plot(kind='barh', figsize=(10,4))
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

# sorting model probabilites to pull related chyrons 
def sort_probs(model, X, y):
    y_net = [1 if net == 'MSNBCW' else 0 for net in y]
    probs = model.predict_proba(X)
    df = pd.DataFrame(probs)
    df['net'] = y_net
    df['index'] = X.index
    msnbc = df.loc[df['net'] == 1]
    fox = df.loc[df['net'] == 0]

    return msnbc, fox

def sort_chyrons(msnbc, fox):
    sorted_ms = []
    sorted_fox = []
    ms_sort = msnbc.sort_values(by=1)
    fox_sort = fox.sort_values(by=1)

    for i in ms_sort['index']:
        sorted_ms.append(chy_summer.iloc[i, 6])

    for i in fox_sort['index']:
        sorted_fox.append(chy_summer.iloc[i, 6])
    
    return sorted_fox, sorted_ms