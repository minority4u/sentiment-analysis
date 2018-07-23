from thesis.Data import Data_cleaner, Data_loader, Data_preparer
from thesis.Vectorizer import TFIDF_Vectorizer, COUNT_Vectorizer, WORD2VEC_Vectorizer
from thesis.Classifier import CNN, LinearSVM, NaiveBayes_sklearn
import thesis.Vectorizer as v
from thesis.Data import Data_cleaner, Data_loader, Data_preparer

import os
import sys
import logging
import thesis.my_logger



class Hypothese(object):

    fitness = 0
    name = 'Hypothese'
    classifier = None
    vectorizer = None
    data_cleaner = None
    data_loader = None

    # define standard parameter assigning to convention over configuration
    def __init__(self, data_loader=None, samples=1000, red_method='tsne', vectorizer='word2vec', w2v_dim = 300):


        if data_loader==None:
            self.data_loader = Data_loader()
        else:
            self.data_loader = data_loader

        self.num_of_samples = samples
        self.red_method = red_method
        self.w2v_dim = w2v_dim
        self.vectorizer = vectorizer

        # initial variant
        self.classifier = LinearSVM()



    def run(self):
        self.vectorizer = v.get_Vectorizer(
                                           vectorizer=self.vectorizer,
                                           num_of_samples=self.num_of_samples,
                                           reduction_methode=self.red_method,
                                           w2v_dimension=self.w2v_dim)



        # dependency injection for the provided data
        data_vectorized = self.vectorizer.vectorize(self.data_loader.get_data())

        # reduce the dimensionality of the training and testing data with tsne
        # no effort, acc 50 - 60 %
        # data_vectorized['x_train_v'] = v.reduce_with_TSNE_single(unreduced_data=data_vectorized['x_train_v'])
        # data_vectorized['x_test_v'] = v.reduce_with_TSNE_single(unreduced_data=data_vectorized['x_test_v'])


        self.classifier.classify(data_vectorized)

        self.classifier.predict(data_vectorized)









    def calc_fitness(self):
        pass

    def mutate(self):
        pass
    def compare_to(self):
        pass


def hypothese_test():
    hyp_test = Hypothese(vectorizer='tfidf')
    hyp_test.run()

if __name__ == "__main__":
    hypothese_test()
    pass