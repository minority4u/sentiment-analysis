# import modules & set up logging
from random import shuffle
import random
from nltk.tokenize import RegexpTokenizer
import gensim, logging
import sys
import os

import thesis.Visualization as vis



# define a logfile and a level at one point to reuse it in all modules
# actually we log to std out and to /logs/SA.log
from gensim import utils
from gensim.models.doc2vec import LabeledSentence, TaggedDocument, Doc2Vec

logger = logging.getLogger()
hdlr = logging.FileHandler('./logs/' + 'word2vec' + '.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
# console output
hdlr_console = logging.StreamHandler()
hdlr_console.setFormatter(formatter)
logger.addHandler(hdlr_console)

logger.setLevel(logging.INFO)





def create_word2vec_model():


    tokenizer = RegexpTokenizer(r'\w+')

    # use relative paths
    os.chdir(os.path.dirname(__file__))
    print(os.getcwd())


    class MySentences(object):
        def __init__(self, dirname):
            self.dirname = dirname
            self.files =['feat.neg','feat.pos']
        def __iter__(self):
            for fname in self.files:
                for line in open(os.path.join(self.dirname, fname)):
                    yield tokenizer.tokenize(line)

    sentences = MySentences('./prepared_data')  # a memory-friendly iterator

    # explain
    # https://radimrehurek.com/gensim/models/word2vec.html
    model = gensim.models.Word2Vec(sentences, sg=0,alpha=0.05, min_count=5, window=10, size=150, sample=1e-2, negative=25, workers=7,)


    for epoch in range(25):
        logger.info('Epoch %d' % epoch)
        model.train(sentences, total_examples=model.corpus_count, epochs=1)

    # store w2v model
    dir = './w2v_model/150_dimensions/word_tokenized/cbow_25/'
    ensure_dir(dir)
    model.save(dir + 'own.d2v')

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def use_word2vec():
    #use_word2vec_with_wordlists()
    pass


def create_doc2vec_model():



    # use relative paths
    os.chdir(os.path.dirname(__file__))
    print(os.getcwd())

    class LabeledLineSentence(object):
        def __init__(self, sources):
            self.sources = sources

            flipped = {}

            # make sure that keys are unique
            for key, value in sources.items():
                if value not in flipped:
                    flipped[value] = [key]
                else:
                    raise Exception('Non-unique prefix encountered')

        def __iter__(self):
            for source, prefix in self.sources.items():
                with utils.smart_open(source) as fin:
                    for item_no, line in enumerate(fin):
                        yield TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

        def to_array(self):
            self.sentences = []
            for source, prefix in self.sources.items():
                with utils.smart_open(source) as fin:
                    for item_no, line in enumerate(fin):
                        self.sentences.append(
                            TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
            return self.sentences

        def sentences_perm(self):
            shuffle(self.sentences)
            return self.sentences

    sources = {'./prepared_data/feat.pos': 'FEAT_POS', './prepared_data/feat.neg': 'FEAT_NEG'}

    sentences = LabeledLineSentence(sources)  # a memory-friendly iterator


    # explain
    # https://radimrehurek.com/gensim/models/word2vec.html
    model = gensim.models.Doc2Vec( min_count=1, window=10, size=300, sample=1e-4, negative=5, workers=7)
    model.build_vocab(sentences.to_array())



    for epoch in range(50):
        logger.info('Epoch %d' % epoch)
        model.train(sentences.sentences_perm(), total_examples=model.corpus_count, epochs=1)

    model.save('./d2v_model/300_dimensions/own.d2v')


def use_Doc2Vec():
    model = Doc2Vec.load('./d2v_model/300_dimensions/own.d2v')

    print(model.docvecs)
    # 10058




if __name__ == "__main__":
    # testing ourpose
    create_word2vec_model()
    #create_doc2vec_model()
    #use_word2vec()
    #use_Doc2Vec()