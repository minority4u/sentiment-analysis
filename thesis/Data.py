# general helper libs
import io
import os
import csv
from random import shuffle
import itertools
import time
import logging
import thesis.my_logger


import Const_params as c
from itertools import islice
import os
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.tokenize import RegexpTokenizer, word_tokenize

from sklearn.model_selection import train_test_split


class Data_preparer(object):
    name = 'Data preparing'

    def __init__(self):
        # change logfile location for the preparing methods
        logger = logging.getLogger()
        hdlr = logging.FileHandler('./logs/' + self.name + '.log')
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr)
        pass

    # count the lines of that file
    def file_len(self, fname):
        with open(fname, encoding='utf-8', errors='ignore') as f:
            for i, l in enumerate(f):
                pass
        return i + 1

    def concat_all_txt_files_in_path(self, source_path, dir_path, sentiment):

        conclusion_file = dir_path + 'feat.' + sentiment
        f_writer = open(conclusion_file, 'a')
        i = 0

        for file in os.listdir(self, source_path):
            if file.endswith(".txt"):
                with open(source_path + file, encoding='latin-1') as f:
                    for line in f:
                        f_writer.write(line + '\n')
                        i += 1
        f_writer.close()
        logging.info('file: ' + conclusion_file + '\n',
              'lines written: ' + str(i) + '\n',
              'current lines in file: ' + str(self.file_len(conclusion_file)) + '\n')

    def split_caggle_data_source_in_pos_and_neg(source_path, dir_path):

        pos_writer = open(dir_path + "feat.pos", 'a')
        neg_writer = open(dir_path + "feat.neg", 'a')
        with open(source_path, encoding='latin-1') as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                # print(row)
                if row[0] == '1':
                    pos_writer.write(row[1] + '\n')
                elif row[0] == '0':
                    neg_writer.write(row[1] + '\n')
                else:
                    logging.info('failure with row: ', row)

    def concat_all_datasources(self, source_paths):
        feats = []
        for path in source_paths:
            with open(path, encoding='utf-8', errors='ignore') as f:
                for line in f:
                    feats.append(line)
                    # print(line)

        shuffle(feats)
        return feats

    def write_list_to_file(self, list, dir_path):
        file = open(dir_path, 'w')
        for item in list:
            file.write(item)
            # print(item)
        file.close()

    def main(self):
        yourpath = './'

        parent = os.path.abspath(os.path.join(yourpath, os.pardir))
        logging.info(parent)

        dir_path = parent + '/data/prepared_data/'
        logging.info(dir_path)


        data_dir = parent + '/data/raw_data/'
        logging.info(data_dir)

        pos_source_paths = []
        neg_source_paths = []

        pos_source_paths.extend([data_dir + 'rt-polaritydata/rt-polarity.pos',
                                 data_dir + 'aclImdb_v1/aclImdb/feat.pos',
                                 data_dir + 'caggle/feat.pos'])
        neg_source_paths.extend([data_dir + 'rt-polaritydata/rt-polarity.neg',
                                 data_dir + 'aclImdb_v1/aclImdb/feat.neg',
                                 data_dir + 'caggle/feat.neg'])
        print('pos: ', [self.file_len(path) for path in pos_source_paths])
        print('neg: ', [self.file_len(path) for path in neg_source_paths])

        for path in pos_source_paths:
            logging.info(path)
            logging.info(self.file_len(path))

        for path in neg_source_paths:
            logging.info(path)
            logging.info(self.file_len(path))

        neg = self.concat_all_datasources(neg_source_paths)
        pos = self.concat_all_datasources(pos_source_paths)


        #print(len(neg) , len(pos))
        logging.info('neg' + str(len(neg)))
        logging.info('pos' + str(len(pos)))

        self.write_list_to_file(neg, dir_path + 'feat.neg')
        self.write_list_to_file(pos, dir_path + 'feat.pos')

        logging.info('length of neg source file: '+ str(self.file_len(dir_path + 'feat.neg')))
        logging.info('length of pos source file: '+ str(self.file_len(dir_path + 'feat.pos')))


        # source_path = "./data/caggle/caggle_sentiment_training_data.txt"
        # dir_path = "./data/caggle/"
        #
        # logging.info('length of source file: ', file_len(source_path))
        #
        # split_caggle_data_source_in_pos_and_neg(source_path, dir_path)
        # logging.info('length of the prepared datafiles: ', file_len(dir_path + 'feat.neg') , file_len(dir_path + 'feat.pos'))

    def split_datafile_into_training_and_testing(self, path):
        train_feats = []
        test_feats = []
        lines = self.file_len(path)
        testline = lines * 0.125
        logging.info(testline)
        trainlines = lines * 0.875
        logging.info(trainlines)
        currentline = 0
        with open(path, encoding='latin-1') as f:
            for line in f:
                if currentline < trainlines:
                    train_feats.append(line)
                else:
                    test_feats.append(line)
                currentline += 1

        self.write_list_to_file(train_feats, 'train_out.csv')
        self.write_list_to_file(test_feats, 'test_out.csv')


class Data_loader(object):
    data = None
    data_preparing = None

    def __init__(self):
        self.__load_data__()
        #self.data_preparing = Data_preparer()
        logging.info('data-loader created')


    """
    self.data =   {'x_train': x_train,
                  'y_train': y_train,
                  'x_test': x_test,
                  'y_test': y_test}
    """
    def __load_data__(self):
        start_time = time.time()
        #negfeats, posfeats = self.load_negative_and_positive_data(data_dir=c.DIR_TO_ROTTEN_DATA)
        negfeats, posfeats = self.load_negative_and_positive_data()

        logging.info("feats loaded - %6.2f seconds ---" % (time.time() - start_time))

        # statistical infos to the loaded data
        self.calculate_statistics_for_this_corpus(negfeats, posfeats)

        ## create labels for our Data
        # tried it with 0 and 1
        # tried it with -1 and 1
        start_time = time.time()
        neg_labels = []
        pos_labels = []
        for _ in negfeats:
            neg_labels.append(c.NEGATIVE)
        for _ in posfeats:
            pos_labels.append(c.POSITIVE)


        # split data into testing and training set + shuffle
        x_train, x_test, y_train, y_test = train_test_split(
            negfeats + posfeats, neg_labels + pos_labels, test_size=0.2, random_state=42)
        logging.info(" feats splitted in training and testing data - %6.2f seconds ---" % (time.time() - start_time))
        print('first training sentence = ' + x_train[0])

        self.data = {'x_train': x_train,
                  'y_train': y_train,
                  'x_test': x_test,
                  'y_test': y_test}



    """
    :param no parameters
    :returns {'x_train': x_train,
                  'y_train': y_train,
                  'x_test': x_test,
                  'y_test': y_test}
    """
    # public methode to receive the loaded data
    def get_data(self):
        return self.data

    # returns the sentence as dictionary
    def sentence_feats(self, line):
        return dict(line, True)


    # returns every word of a sentence as dic
    def word_feats(self, words):
        return dict([(word, True) for word in words])


        # returns every word of a sentence as dic added stopword filtering
        # adding stemming will decease the acc


    def stopword_filtered_word_feats_old(self, words):
        filtered_dic = {}
        stopset = set(stopwords.words('english'))
        # ps = PorterStemmer()
        # for word in words:
        #     word_stemmed = ps.stem(word)
        #     if word_stemmed not in stopset:
        #         filtered_dic[word_stemmed] = True
        # return filtered_dic
        return dict([(word, True) for word in words if word not in stopset])

        # returns every word of a sentence as dic added stopword filtering
        # adding stemming will decease the acc
        # without a dictionary


    def stopword_filtered_word_feats(self, words):
        filtered_dic = {}
        stopset = set(stopwords.words('english'))
        # ps = PorterStemmer()
        # for word in words:
        #     word_stemmed = ps.stem(word)
        #     if word_stemmed not in stopset:
        #         filtered_dic[word_stemmed] = True
        # return filtered_dic
        return [word for word in words if word not in stopset]




        # create bigrams for each sentence, included stopword filtering
        # we can decide by BigramAssocMeasures.chi_sp how many of the best bigrams we use


    def bigram_word_feats(self, words, score_fn=BigramAssocMeasures.chi_sq, n=c.NUMBER_OF_BIGRAMS_TO_KEEP):
        # filter the stopwords to raise the performance, but loose bigram semantic -> worse acc
        words = self.stopword_filtered_word_feats(words)
        # create bigrams
        bigram_finder = BigramCollocationFinder.from_words(words)
        # select only the n best bigrams by chi_sq
        bigrams = bigram_finder.nbest(score_fn, n)

        return dict([(ngram, True) for ngram in itertools.chain(words, bigrams)])


    def loadDataFromDirecory(self, featx, source_directories):
        # load data from a given directory
        # we have to tokenize every line, because the movie_review words also returns tokenized words
        negfeats = []
        posfeats = []
        # remove dots by regex-tokenizer
        tokenizer = RegexpTokenizer(r'\w+')

        # read the data and create tuples which looks like this:
        # ({token1:TRUE, token2:TRUE}, pos)
        with open(source_directories[0], encoding='latin-1') as f_neg:
            for line in f_neg:
                negfeats.append([featx(tokenizer.tokenize(line)), 'neg'])

        with open(source_directories[1], encoding='latin-1') as f_pos:
            for line in f_pos:
                posfeats.append([featx(tokenizer.tokenize(line)), 'pos'])
        return negfeats, posfeats



    def load_negative_and_positive_data(self, data_dir=c.DIR_TO_PREPARED_DATA):
        # use relative paths
        os.chdir(os.path.dirname(__file__))
        print(os.getcwd())

        # read the first n lines of the sentiment data and returns two lists
        with open(data_dir[0], encoding='latin-1', errors='replace') as f_neg:
            negfeats = list(islice(f_neg, 0, c.AMOUNT_OF_DATAFEATS_TO_LOAD))

            if c.CLEANING:
                logging.info('cleaning negative feats')
                negfeats = [review_to_words(negfeat) for negfeat in negfeats]
                logging.info('cleaning negative feats done')


        with open(data_dir[1], encoding='latin-1', errors='replace') as f_pos:
            posfeats = list(islice(f_pos, 0, c.AMOUNT_OF_DATAFEATS_TO_LOAD))
            if c.CLEANING:
                logging.info('cleaning positive feats')
                posfeats = list(review_to_words(posfeat) for posfeat in posfeats)
                logging.info('cleaning positive feats done')

        return negfeats, posfeats

    def calculate_statistics_for_this_corpus(self, negfeats, posfeats):



        number_of_neg_feats = len(negfeats)
        number_of_pos_feats = len(posfeats)

        logging.info('neg:' + str(len(negfeats)) + ' pos: ' + str(len(posfeats)) + ' loaded')





        # words_in_negfeats = [word_tokenize(feat) for feat in negfeats]
        # words_in_posfeats = [word_tokenize(feat) for feat in posfeats]
        # average_number_of_words_per_feat_negative = len(words_in_negfeats) / number_of_neg_feats
        # average_number_of_words_per_feat_positive = len(words_in_posfeats) / number_of_pos_feats
        #
        # logging.info('neg feats: ' + str(number_of_neg_feats))
        # logging.info('pos feats: ' + str(number_of_pos_feats))
        # logging.info('words in negative feats: ' + str(len(words_in_negfeats)))
        # logging.info('words in positive feats: ' + str(len(words_in_posfeats)))
        # logging.info('average number of words - negative feats: ' + str(average_number_of_words_per_feat_negative))
        # logging.info('average number of words - positive feats:' + str(average_number_of_words_per_feat_positive))


    def load_negative_and_positive_data_as_dic(self, data_dir=c.DIR_TO_PREPARED_DATA):
        # use relative paths
        os.chdir(os.path.dirname(__file__))
        print(os.getcwd())

        # read the first n lines of the sentiment data and returns two lists
        with open(data_dir[0], encoding='latin-1') as f_neg:
            negfeats = list(islice(f_neg, 0, c.AMOUNT_OF_DATAFEATS_TO_LOAD))

        with open(data_dir[1], encoding='latin-1') as f_pos:
            posfeats = list(islice(f_pos, 0, c.AMOUNT_OF_DATAFEATS_TO_LOAD))
        logging.info(len(negfeats), len(posfeats))

        return {'neg': negfeats, 'pos': posfeats}


class Data_cleaner(object):
    def __init__(self, data):
        cleaned_data= {}

        #
        for key, value in data:
            for review in value:

                cleaned_data[key] = []
                cleaned_data[key].append(review_to_words(review))

        return cleaned_data

    def review_to_words(raw_review):
        # logging.info('cleaning data')
        # Function to convert a raw review to a string of words
        # The input is a single string (a raw movie review), and
        # the output is a single string (a preprocessed movie review)
        #
        # 1. Remove HTML
        review_text = BeautifulSoup(raw_review, 'html.parser').get_text()

        # handle negation
        review_text = handle_negation(review_text)

        #
        # 2. Remove non-letters
        letters_only = re.sub("[^a-zA-Z_]", " ", review_text)
        #
        # 3. Convert to lower case, split into individual words
        words = letters_only.lower().split()
        #
        # 4. In Python, searching a set is much faster than searching
        #   a list, so convert the stop words to a set
        stops = set(stopwords.words("english"))
        #
        # 5. Remove stop words
        # meaningful_words = [w for w in words if not w in stops]
        meaningful_words = words
        #
        # 6. Join the words back into one string separated by space,
        # and return the result.
        # logging.info('cleaning data - done')
        return (" ".join(meaningful_words))



def load_neg_pos_wordlist(num_of_words=2000):

    from itertools import islice
    with open('./prepared_data/negative-words.txt', 'rt', encoding='utf8') as f:
        neg_words = list(islice(f, 0, num_of_words))
        neg_words = [line.rstrip('\n') for line in neg_words]


    with open('./prepared_data/positive-words.txt', 'rt', encoding='utf8') as f:
        pos_words = list(islice(f, 0, num_of_words))
        pos_words = [line.rstrip('\n') for line in pos_words]

    return neg_words, pos_words


# returns a pandas data frame with the training data
def load_rotten_tomatos_as_df():

    train_df = pd.read_csv('./prepared_data/raw_data/rotten_tomato/train.tsv', sep='\t', lineterminator='\n')
    test_df = pd.read_csv('./prepared_data/raw_data/rotten_tomato/test.tsv', sep='\t', lineterminator='\n')

    df_all = train_df.append(test_df, ignore_index=True)

    return df_all


def test_load_rotten_tomatos_as_df():
    df = load_rotten_tomatos_as_df()
    print(df.shape)




    # delete neutral
    df = df[df['Sentiment'] !=2]
    print(df.shape)

    # replace mostly negative with negative
    df['Sentiment'] =  df['Sentiment'].replace([1,0],0 )
    print(df.shape)

    # replace mostly positive with positive
    df['Sentiment'] = df['Sentiment'].replace([3,4],1)
    print(df.shape)

    neg_df = df[df['Sentiment'] == 0]
    pos_df = df[df['Sentiment'] == 1]
    print(neg_df)
    neg_df['Phrase'].to_csv('./prepared_data/raw_data/rotten_tomato/feat.neg', index=False)
    pos_df['Phrase'].to_csv('./prepared_data/raw_data/rotten_tomato/feat.pos', index=False)
   # df[].to_csv('./prepared_data/raw_data/rotten_tomato/feat.neg')

    #print(df['Sentiment'])


def handle_negation(review):

    reviews = ["I didn't like that movie. But I love the actors."]

    negations = ["n't, not, never"]

    negation_handled_review =[]

    from nltk.tokenize import sent_tokenize
    print('raw')
    print(review)

    sent_tokenize_list = sent_tokenize(review)

    for i, sent in enumerate(sent_tokenize_list):
        negation = False
        print('sentence: ' + str(i))
        print(sent)

        for word in word_tokenize(sent):
            if word in negations:
                negation = not negation
            if negation:
                print('NOT_' + word)
                negation_handled_review
            else:
                print(word)


# manuel function to clean a review
# This could be done by regex tokenizer or word tokenizer from nltk
from bs4 import BeautifulSoup
import re

def review_to_words( raw_review ):
    #logging.info('cleaning data')
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review,'html.parser').get_text()

    #
    # 2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z_]", " ", review_text)
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))
    #
    # 5. Remove stop words
    #meaningful_words = [w for w in words if not w in stops]
    meaningful_words = words
    #
    # 6. Join the words back into one string separated by space,
    # and return the result.
    #logging.info('cleaning data - done')
    return( " ".join( meaningful_words ))







def test_data_loader():
    dl = Data_loader()
    data = dl.get_data()

    train = data['x_train'][0]
    #print(train)

    handle_negation([train])


def test_data_preparer():
    dp = Data_preparer()
    dp.main()



if __name__ == "__main__":
    test_data_loader()
    #test_data_preparer()
    #test_load_rotten_tomatos_as_df()