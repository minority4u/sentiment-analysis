from thesis.Hypothese import Hypothese
from thesis.Data import Data_cleaner, Data_loader, Data_preparer

import logging
import os
import sys
import thesis.my_logger



class Ml_operator(object):
    def __init__(self, name):
        self.name = name




class Evolution_alg(object):

    population = []
    population_size = 1
    name = 'SA'


    def __init__(self):

        # set up a unique logger
        logging.info(' start ------------------------------------------------------------------')






        # possible variants
        # self.vectorizer = ['word2vec', 'tfidf']
        # self.samples = [1000] # n examples possible
        # self.methods = ['pca', 'tsne']
        # self.dim = [50, 100, 300]




        self.vectorizer = ['word2vec']
        self.num_of_samples_to_print = [500]
        self.methods = ['tsne']
        self.dim = [300]
        self.preprocessing = ['stopwords']


        self.Data_loader = Data_loader()
        #self.Data_cleaner = Data_cleaner()

        # initialize the population with hypotheses to train/evaluate on
        # how many samples should we plot
        for s in self.num_of_samples_to_print:
            # run with each vectorizer defined
            for vec in self.vectorizer:
                for m in self.methods:
                    # only word2vec has different vectorizer models
                    if vec == "word2vec":
                        for d in self.dim:

                            logging.info('initialize w2v hyp')
                            self.population.append(Hypothese(data_loader = self.Data_loader,
                                                             samples=s, red_method=m,
                                                             w2v_dim=d,
                                                             vectorizer=vec))
                    else:
                        logging.info('initialize tfidf hyp')
                        self.population.append(
                            Hypothese(data_loader=self.Data_loader,
                                      samples=s,
                                      red_method=m,
                                      vectorizer=vec))

    def run(self):


        self.testrun()

        logging.info(' end ------------------------------------------------------------------')

        # self.print_Population_overview()
        # self.selection()
        # self.crossover()
        # self.mutate()
        # self.update()
        # self.evaluate_and_sort_population()

    def print_Population_overview(self):
        pass

    def selection(self):
        pass

    def crossover(self):
        pass

    def mutate(self):
        pass

    def update(self):
        pass

    def evaluate_and_sort_population(self):
        pass

    def testrun(self):

        #data = self.Data_loader.load_data()

        for hyp in self.population:
            if type(hyp) is Hypothese:
                hyp.run()



# ML = Ml_operator(name='machine learning alg')
# print(ML.name)
if __name__ == "__main__":
    evolution = Evolution_alg()
    evolution.run()