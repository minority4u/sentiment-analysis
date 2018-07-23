
# parameter for easy classifier evaluation

DIR_TO_POLARITY_DATA = ('./data/rt-polaritydata/rt-polarity.neg', './data/rt-polaritydata/rt-polarity.pos')
DIR_TO_ACLIMDB_DATA = ('./data/aclImdb_v1/aclImdb/feat.neg', './data/aclImdb_v1/aclImdb/feat.pos')
DIR_TO_CAGGLE_DATA = ('./data/caggle/feat.neg', './data/caggle/feat.neg')
DIR_TO_PREPARED_DATA = ('prepared_data/feat.neg', 'prepared_data/feat.pos')
DIR_TO_ROTTEN_DATA = ('prepared_data/raw_data/rotten_tomato/feat.neg', 'prepared_data/raw_data/rotten_tomato/feat.pos')

#CLASSIFIER_DIR_PATH = './classifier/naive_bayes_bigram_classifier.pickle'
#CLASSIFIER_DIR_PATH = './classifier/naive_bayes_stopword_filtered_unshuffeled_classifier.pickle'
#CLASSIFIER_DIR_PATH = './classifier/naive_bayes_stopword_filtered_classifier.pickle'
#CLASSIFIER_DIR_PATH = './classifier/svm_stopword_filtered_classifier.pickle'
#CLASSIFIER_DIR_PATH = './classifier/svm_bigram_stopword_filtered_classifier.pickle'
CLASSIFIER_DIR_PATH = './naive_bayes.pickle'

### set the feat methode
#WORD_FEAT_METHODE = d.stopword_filtered_word_feats
#WORD_FEAT_METHODE = d.bigram_word_feats

### set the classification algorythm
#CLASSIFIER_ALGORYTHM = t.naive_bayes_based_evaluation
#CLASSIFIER_ALGORYTHM = t.lexicon_based_evaluation

# the dataset consists of 31tsd entries
# 2017-05-20 18:26:17,200 INFO length of neg source file: 33422
# 2017-05-20 18:26:17,231 INFO length of pos source file: 34326
#AMOUNT_OF_DATAFEATS_TO_LOAD = 33330
AMOUNT_OF_DATAFEATS_TO_LOAD = 10
PERCENT_PART_FOR_TRAINING = 0.75
NUMBER_OF_BIGRAMS_TO_KEEP = 1000
CONCAT_ALL_DATASOURCES = True
TRAIN_NEW_CLASSIFIER = True
CLEANING = False
# label for the sentiment feats
POSITIVE = 1
NEGATIVE = 0






