
from __future__ import print_function

from pprint import pprint
from time import time
import logging

import nltk
import numpy as np
from nltk import RegexpTokenizer
from sklearn import svm, feature_selection

from sklearn.datasets import fetch_20newsgroups
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import SGDClassifier
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from nltk.corpus import stopwords
from sklearn.svm import LinearSVC

from thesis.Data import Data_loader as dl

print(__doc__)

# define a logfile and a level at one point to reuse it in all modules
# actually we log to std out and to /logs/SA.log

logger = logging.getLogger()
hdlr = logging.FileHandler('./logs/' + 'Gridsearch' + '.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
# console output
hdlr_console = logging.StreamHandler()
hdlr_console.setFormatter(formatter)
logger.addHandler(hdlr_console)
logger.setLevel(logging.INFO)

from thesis.Data import Data_loader
data = Data_loader().get_data()
#tokenizer = RegexpTokenizer(r'\w+')
tokenizer = nltk.TweetTokenizer()

def tokenize(text):
    #tokens = nltk.word_tokenize(text)
    tokens = tokenizer.tokenize(text)
    stems = []
    for item in tokens:
        stems.append(nltk.PorterStemmer().stem(item))
        #stems.append(item)
    return stems




# initial pipeline
# pipeline = Pipeline([
#     ('vect', CountVectorizer()),
#     ('clf', svm.LinearSVC()),
# ])

# svm pipeline with best parameters choosen - working pipeline
# dont touch
# pipeline = Pipeline([
#     ('vect', CountVectorizer(ngram_range=(1, 2), max_df=0.8, min_df=2, analyzer='word', stop_words=None, max_features=None, binary=True)),
#     ('tfidf', TfidfTransformer(sublinear_tf=True, use_idf=True, norm='l2', smooth_idf=False)),
#     ('clf', svm.LinearSVC(C=1.0, tol=0.1, loss='squared_hinge'))
# ])

# Different feature selection mechanism, accuracy could not be improved
#cla = ExtraTreesClassifier()
#cla = LinearSVC()
cla = SelectKBest(chi2, k=1000)
# tokenize methode could be changed to own tokenize method
pipeline = Pipeline([

    ('vect', CountVectorizer(ngram_range=(1, 2), max_df=0.8, min_df=2,  analyzer=tokenize, stop_words=None, max_features=None, binary=True)),
    #('tfidf', TfidfTransformer(sublinear_tf=True, use_idf=True, norm='l2', smooth_idf=False)),
    #('feature_selection', SelectFromModel(cla)),
    #('chi', SelectKBest(chi2, k=5000)),
    #('clf', svm.LinearSVC(loss='squared_hinge', C=0.9, tol=0.1))
    ('clf', BernoulliNB())
])


# define test Pipeline
# pipeline with TSNE
# pipeline = Pipeline([
#     ('vect', CountVectorizer(ngram_range=(1, 2), max_df=0.75, min_df=2, analyzer='word', binary=True)),
#     ('tfidf', TfidfTransformer(sublinear_tf=True, use_idf=True)),
#     #('tsne'), TSNE(n_components=2, random_state=0),
#     ('clf', svm.LinearSVC()),
# ])

# AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
#                          algorithm="SAMME",
#                          n_estimators=200))

# parameters = {
#     'vect__max_df': (0.70, 0.75, 0.8), # 0,8 is better than 0.5, 0.9 or 0.5, with full dataset 0.75 is better.
#     'vect__max_features': (None, 10000), # None is better than 5000, 10000, 50000
#     'vect__ngram_range': ((1, 2), (1,1), (1,3), (2,2), (2,3)),  # bigrams are better
#     'vect__binary': (True, False), # true is better
#     'vect__min_df': (2, 5), # 2 is better than 5
#     'vect__analyzer': ('word', 'char', 'char_wb', tokenize), # word analyzer works better thatn char or char_wb tokenize method is worse
#     'vect__stop_words': ('english', None) # no stopwords works better than with the usage of stopwords
#     #'tfidf__sublinear_tf': (True, False), # true is better
#     #'tfidf__use_idf': (True, False), # True is better
#     #'tfidf__norm': ('l1', 'l2'),
#     #'clf__C': (1.0, 2.0), # 1 is better
#     #'clf__tol': (1e-2, 1e-3), # 1e-2 is better
#     #'clf__penalty': ('l2', 'elasticnet'),
#     #'clf__max_iter': (1000, 10000, 50000)
# }

Cs = [1e-2, 1, 1e2]
Tol = [1e-2,1e-1, 1, 1e1]

# parameters for linear svm
parameters = {
    #'vect__max_df': (0.70, 0.75, 0.8), # 0,8 is better than 0.5, 0.9 or 0.5, with full dataset 0.75 is better.
    #'vect__max_features': (None, 10000), # None is better than 5000, 10000, 50000
    #'vect__ngram_range': ((1, 2), (1,1), (1,3)),  # bigrams are better
    #'vect__binary': (True, False), # true is better
    #'vect__min_df': (3,2,1), # 2 is better than 5, or 0.1
    #'tfidf__sublinear_tf': (True, False), # True is better
    #'tfidf__use_idf': (True, False), # True is better
    #'tfidf__smooth_idf': (True, False), # False is better
    #'tfidf__norm': ('l1', 'l2', None), # l2
    #'tfidf__norm': ('l1', 'l2', None),
    #'clf__C': (0.8,0.9,1.0), # 1 is better
    #'clf__tol': (1e-1, 1e-2, 1e-3, 1e-4), # 0.1 is better
    #'clf__loss': ('squared_hinge', 'hinge'), # squared hinge is better
    #'clf__penalty': ('l2', 'l1'),
}

if __name__ == "__main__":
    # print all possible parameters
    for elem in pipeline.steps:
        print(elem[1])
        print(elem[1].get_params().keys())




    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    # find the best parameters for both the feature extraction and the
    # classifier
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=3, cv=4)

    # plot loss and C accuracy
    #grid_search = GridSearchCV(pipeline, dict(clf__C=Cs, clf__tol=Tol), n_jobs=-1, verbose=1, cv=2)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    print(parameters)
    t0 = time()
    grid_search.fit(data['x_train'], data['y_train'])
    print("done in %0.3fs" % (time() - t0))
    print()

    # # https://stackoverflow.com/questions/37161563/how-to-graph-grid-scores-from-gridsearchcv
    # import matplotlib.pyplot as plt
    #
    # scores = [x[1] for x in grid_search.grid_scores_]
    # scores = np.array(scores).reshape(len(Cs), len(Tol))
    #
    # for ind, i in enumerate(Cs):
    #     plt.plot(Tol, scores[ind], label='C: ' + str(i))
    # plt.legend()
    # plt.xlabel('Los')
    # plt.ylabel('Mean score')
    # plt.show()




    import thesis.Visualization as plotter

    # plot the most informative features of the best pipeline
    features = grid_search.best_estimator_.named_steps['vect'].get_feature_names()
    logging.info(features[0])
    logging.info(len(features))
    clf = grid_search.best_estimator_.named_steps['clf']
    plotter.plot_coefficients(clf, features,fname='test')

    # show best accuracy from the 4 fold cross validation with the validation data
    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    # print classification_report with the unseen testing data
    clf = grid_search.best_estimator_
    prediction = clf.predict(data['x_test'])
    target_names = ['negative', 'positive']
    print(classification_report(data['y_test'], prediction, target_names=target_names))