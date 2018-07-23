from numpy import matrix
from sklearn import svm, preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

import thesis.Visualization as plotter

import time
import os
import sys
import logging
import thesis.my_logger
import collections

import thesis.IO_Organizer as io



# --- NLTK libs --- #
import nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk import precision, RegexpTokenizer
from nltk import recall




class RandomForest(object):
    pass

class LinearSVM(object):
    Classifier_liblinear = None
    prediction_liblinear = None
    name = 'linearSVM'

    def __init__(self, C=1, tol=0.0001):
        # create a linear svm
        self.Classifier_liblinear = svm.LinearSVC(C=C, tol=tol)
        #self.Classifier_liblinear = SVC(kernel='rbf')
        self.scaler = preprocessing.StandardScaler()



    def classify(self, data_vectorized):

        logging.info(self.name + ' new run ---------------------------')

        #plot_learning(data_vectorized)

        logging.info("training new model ... ")
        start_time = time.time()


        # Perform training with linear SVM

        #x_train_v_scaled = self.scaler.fit_transform(data_vectorized['x_train_v'])
        x_train_v_scaled = data_vectorized['x_train_v']
        self.Classifier_liblinear.fit(x_train_v_scaled  , data_vectorized['y_train'])
        self.time_training = (time.time() - start_time)
        logging.info("model trained - %6.2f seconds " % self.time_training)

        start_time = time.time()
        #io.save_classifier(self.Classifier_liblinear)
        self.time_saving = (time.time() - start_time)
        logging.info("model saved to file - %6.2f seconds " % self.time_saving)




    def predict(self, data_vectorized):
        target_names = ['negative', 'positive']
        #x_test_v_scaled = self.scaler.fit_transform(data_vectorized['x_test_v'])
        x_test_v_scaled = data_vectorized['x_test_v']
        start_time = time.time()
        self.prediction_liblinear = self.Classifier_liblinear.predict(x_test_v_scaled)
        self.time_prediction = (time.time() - start_time)
        logging.info("prediction finished - %6.2f seconds " % self.time_prediction)



        # cross validation
        # logging.info("cross validation ... ")
        # start_time = time.time()
        # scores = cross_val_score(self.Classifier_liblinear,
        #                          data_vectorized['x_train_v'],
        #                          data_vectorized['y_train'],
        #                          cv=3, n_jobs=-1)
        #
        # logging.info("Cross-Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        # logging.info("Cross-Validation finished- %6.2f seconds " % (time.time() - start_time))

        # # # Print results in a nice table linearSVC
        logging.info("Results for LinearSVC()")
        logging.info("Training time: %fs; Prediction time: %fs" % (self.time_training, self.time_prediction))
        logging.info(classification_report(data_vectorized['y_test'], self.prediction_liblinear, target_names=target_names))

        # ### plot top features - only possible for linear and tfidf
        try:
            plotter.plot_coefficients(self.Classifier_liblinear, data_vectorized['vectorizer'].get_feature_names(), fname=self.name)
        except:
            logging.info('feature-plotting not possible')

        io.save_classifier(self.Classifier_liblinear)




class NaiveBayes_sklearn(object):
    name ='NaiveBayesClassifier_sklearn'
    classifier = None


    def __init__(self):
        from sklearn.naive_bayes import MultinomialNB
        self.classifier = MultinomialNB()
        pass

    def word_feats(self, words):
        return dict([(word, True) for word in words])


    def classify(self, data_vectorized):


        logging.info(self.name + ' new run ---------------------------')
        start_time = time.time()
        # plot_learning(data_vectorized)

        train_features = data_vectorized['x_train_v']

        # Fit a naive bayes model to the training data.
        # This will train the model using the word counts we compute, and the existing classifications in the training set.


        self.classifier.fit(train_features, [int(r) for r in data_vectorized['y_train']])
        self.time_training = (time.time() - start_time)





    def predict(self, data_vectorized):
        from sklearn import metrics
        start_time = time.time()
        target_names = ['negative', 'positive']

        # Now we can use the model to predict classifications for our test features.
        #
        self.predictions = self.classifier.predict(data_vectorized['x_test_v'])

        actual = [int(r) for r in data_vectorized['y_test']]

        self.time_prediction = (time.time() - start_time)

        # Compute the error.  It is slightly different from our model because the internals of this process work differently from our implementation.
        fpr, tpr, thresholds = metrics.roc_curve(actual, self.predictions, pos_label=1)
        print("Multinomial naive bayes AUC: {0}".format(metrics.auc(fpr, tpr)))
        # # # Print results in a nice table linearSVC
        logging.info("Results for " + self.name)
        logging.info("Training time: %fs; Prediction time: %fs" % (self.time_training, self.time_prediction))
        logging.info(
        classification_report(data_vectorized['y_test'], self.predictions, target_names=target_names))

import itertools
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures


class NaiveBayes_nltk(object):
    name = 'NaiveBayesClassifier_nltk'
    classifier = None
    tokenizer = name

    def __init__(self):
        # self.classifier = NaiveBayesClassifier()
        self.tokenizer = RegexpTokenizer(r'\w+')
        pass

    def word_feats(self, words):
        return dict([(word, True) for word in words])

    def bigram_word_feats(self, words, score_fn=BigramAssocMeasures.chi_sq, n=1000):
        from nltk.corpus import stopwords

        # stopword filtering has worse acc than without
        stopset = set(stopwords.words('english'))
        # stopword filtering
        #words = [word for word in words if word not in stopset]

        bigram_finder = BigramCollocationFinder.from_words(words)
        bigrams = bigram_finder.nbest(score_fn, n)
        return dict([(ngram, True) for ngram in itertools.chain(words, bigrams)])


    def classify(self, data_vectorized):
        target_names = ['negative', 'positive']
        start_time = time.time()
        import collections
        import nltk.metrics
        from nltk.classify import NaiveBayesClassifier
        from nltk.corpus import movie_reviews
        logging.info(self.name + ' new run ---------------------------')
        logging.info('split')
        # transform the training feats in the format nltk asks for
        # use the non vectorized words
        trainfeats = []
        for i, feat in enumerate(data_vectorized['x_train']):
            # set the feat and the label
            # tokenize with the bigram methode
            feat_as_words = self.bigram_word_feats(self.tokenizer.tokenize(feat))
            label = data_vectorized['y_train'][i]
            trainfeats.append((feat_as_words, label))
        logging.info('word feats created')
        logging.info('training the classifier')

        # train the naive bayes
        self.classifier = NaiveBayesClassifier.train(trainfeats)
        self.time_training = (time.time() - start_time)


    def predict(self, data_vectorized):
        start_time = time.time()

        # format the testfeats in the format nltk asks for
        # use the word without vectorizing
        testfeats = []
        logging.info('create the testing feats')
        for i, feat in enumerate(data_vectorized['x_test']):

            feat_as_words = self.bigram_word_feats(self.tokenizer.tokenize(feat))
            label = data_vectorized['y_test'][i]
            testfeats.append((feat_as_words, label))


        refsets = collections.defaultdict(set)
        testsets = collections.defaultdict(set)
        for i, (feats, label) in enumerate(testfeats):
            refsets[label].add(i)
            observed = self.classifier.classify(feats)
            testsets[observed].add(i)
        self.time_prediction = (time.time() - start_time)


        logging.info("Results for" + self.name + "with nltk scoring methods")
        logging.info("Training time: %fs; Prediction time: %fs" % (self.time_training, self.time_prediction))
        # logging.info(
        #     classification_report(data_vectorized['y_test'], self.predictions, target_names=target_names))

        logging.info('--- accuracy: %6.2f ---' % nltk.classify.util.accuracy(self.classifier, testfeats))
        logging.info('--- pos precision: %6.2f ---' % precision(refsets[1], testsets[1]))
        logging.info('--- pos recall: %6.2f ---' % recall(refsets[1], testsets[1]))
        logging.info('--- neg precision: %6.2f ---' % precision(refsets[0], testsets[0]))
        logging.info('--- neg recall: %6.2f ---' % recall(refsets[0], testsets[0]))
        logging.info("--- testing done - %6.2f seconds ---" % (time.time() - start_time))
        logging.info(self.classifier.most_informative_features(n=10))
        self.classifier.show_most_informative_features()



class CNN(object):
    def __init__(self):
        pass
    def classify(self):
        pass



######################## testing purpose ####################

def test_LinearSVM():
    # test svm with tdidf-vectorized data
    from thesis.Data import Data_loader
    import thesis.Vectorizer as vec

    data = Data_loader().get_data()
    vec = vec.get_Vectorizer(vectorizer='tfidf')
    #vec = vec.get_Vectorizer(vectorizer='word2vec')
    clf = LinearSVM()

    vectorized_data = vec.vectorize(data=data)
    clf.classify(vectorized_data)
    clf.predict(vectorized_data)

def test_NaiveBayes_sklearn():
    from thesis.Data import Data_loader
    import thesis.Vectorizer as vec

    # load data
    data = Data_loader().get_data()
    # create a vectorizer
    tfidf_vec = vec.get_Vectorizer(vectorizer='tfidf')
    # create a classifier
    clf = NaiveBayes_sklearn()
    # vectorize the data
    vectorized_data = tfidf_vec.vectorize(data=data)
    # train classifier
    clf.classify(vectorized_data)
    # inverence for the classifier
    clf.predict(vectorized_data)



def test_NaiveBayes_NLTK():
    from thesis.Data import Data_loader
    # load data
    data = Data_loader().get_data()

    clf_nltk = NaiveBayes_nltk()
    # test for NLTK Naive Bayes
    clf_nltk.classify(data_vectorized=data)
    clf_nltk.predict(data_vectorized=data)



if __name__ == "__main__":
    #test_LinearSVM()
    test_NaiveBayes_sklearn()
    #test_NaiveBayes_NLTK()


