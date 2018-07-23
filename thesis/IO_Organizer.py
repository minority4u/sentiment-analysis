import pickle
import os

DIR = 'cached_data/'

def save_classifier(classifier, name=''):
    # get relative path and classifier-name as pickle file
    fname = DIR + classifier.__class__.__name__ + '.pickle'

    fn = os.path.join(os.path.dirname(__file__), fname)
    f = open(fn, 'wb')
    pickle.dump(classifier, f)
    f.close()

def load_classifier(classifier_name):
    # get relative path and classifier-name as pickle file
    fname = DIR + classifier_name + '.pickle'

    fn = os.path.join(os.path.dirname(__file__), fname)
    f = open(fn, 'rb')
    classifier = pickle.load(f)
    f.close()
    return classifier


def save_vectorizer(vectorizer, name=''):
    # get relative path and classifier-name as pickle file
    fname = DIR + vectorizer.__class__.__name__ + '.pk'

    fn = os.path.join(os.path.dirname(__file__), fname)
    with open(fn, 'wb') as fin:
        pickle.dump(vectorizer, fin)

def load_vectorizer(vectorizer_name):
    # get relative path and classifier-name as pickle file
    fname = DIR + vectorizer_name + '.pk'
    fn = os.path.join(os.path.dirname(__file__), fname)
    f = open(fn, 'rb')
    vectorizer = pickle.load(f)
    f.close()
    return vectorizer

def save_features(features, name):
    # get relative path and classifier-name as pickle file
    fname = DIR + features.__class__.__name__ + '_' + name + '.pk'

    fn = os.path.join(os.path.dirname(__file__), fname)
    with open(fn, 'wb') as fin:
        pickle.dump(features, fin)

def load_features(feature_name):
    vectorizer = None
    # get relative path and classifier-name as pickle file
    fname = DIR + feature_name + '.pk'
    fn = os.path.join(os.path.dirname(__file__), fname)
    f = open(fn, 'rb')
    vectorizer = pickle.load(f)
    f.close()

    return vectorizer


# for testing purpose
script_dir = os.path.dirname(__file__)
print(script_dir)