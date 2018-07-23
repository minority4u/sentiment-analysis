import gensim
import numpy as np
from sklearn import preprocessing, decomposition
from gensim.models import Word2Vec
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

import thesis.Visualization as vis
import thesis.Data as data
import thesis.Vectorizer as vec
import thesis.Classifier as cls
from thesis.Data import Data_loader
import Const_params as c

import time
import logging
import random

tsne_singelton_model = None
svd_singelton_model = None
normalizer_singelton_model = None



def use_word2vec_with_wordlists():
    # define general testing parameters for word2vec plotting
    words_to_load = 2000
    # define the min difference between the neg and pos averaged wordvectors
    bias = 0.4
    # tsne related params
    perplexity = 150
    learning_rate = 1000
    # reduce by tsne or pca
    reduction_methode = 'pca'
    # filter the most significant dimensions

    extract_dim = True
    normalize = True
    truncate_by_svd=True

    neg_v = []
    pos_v = []
    extracted_neg_wordvectors = []
    extracted_pos_wordvectors = []



    model = Word2Vec.load('./w2v_model/300_dimensions/word_tokenized/own.d2v')
    mod = model.wv
    del model

    #mod = gensim.models.KeyedVectors.load_word2vec_format('./w2v_model/GoogleNews-vectors-negative300.bin',binary=True )

    test_words = {}
    test_words['neg'], test_words['pos'] = data.load_neg_pos_wordlist(num_of_words=words_to_load)


    for word in test_words['neg']:
        try:
            word_vector = mod[word]
            neg_v.append(word_vector)
        except:
            continue

    for word in test_words['pos']:
        try:
            word_vector = mod[word]
            pos_v.append(word_vector)
        except:
            continue



    # avg all neg and pos words for each dimension
    avg_neg = vec.avg_vectors(neg_v)
    avg_pos = vec.avg_vectors(pos_v)
    avgs = []
    avgs.append(avg_neg)
    avgs.append(avg_pos)
    difference = vec.diff(avg_neg, avg_pos, bias=bias)

    # plot each dimensions of our words, the average and the difference
    vis.plot_each_dim(neg_v=neg_v, pos_v=pos_v, avgs=avgs, used_bias=bias, diff=difference, filename='words' )

    ############## plot most informative dimensions ##############
    #plot_sentiment_distribution(neg_v=neg_v, pos_v=pos_v, source='words')

    # extract the significant dimensions of our word vectors according to a defined bias
    if extract_dim:
        relevant_indexes = vec.extraxt_rel_indexes(difference)
        [extracted_neg_wordvectors.append(vec.extract_rel_dim_vec(v, relevant_indexes)) for v in neg_v]
        [extracted_pos_wordvectors.append(vec.extract_rel_dim_vec(v, relevant_indexes)) for v in pos_v]
    else:
        extracted_neg_wordvectors = neg_v
        extracted_pos_wordvectors = pos_v

    # try to classify the words
    # first with all dimensions later with only the most significant dimensions
    neg_labels = []
    pos_labels = []
    for _ in neg_v:
        neg_labels.append(c.NEGATIVE)
    for _ in pos_v:
        pos_labels.append(c.POSITIVE)

    # split data into testing and training set + shuffle
    x_train, x_test, y_train, y_test = train_test_split(
        neg_v + pos_v, neg_labels + pos_labels, test_size=0.25, random_state=42)


    cl = LinearSVC()
    cl.fit(x_train, y_train)
    pred = cl.predict(x_test)
    acc = accuracy_score(y_true=y_test,y_pred= pred)
    logging.info('acc with all dimensions: ' + str(acc))


    # split data into testing and training set + shuffle
    x_train, x_test, y_train, y_test = train_test_split(
        extracted_neg_wordvectors + extracted_pos_wordvectors, neg_labels + pos_labels, test_size=0.25, random_state=42)

    cl = LinearSVC()
    cl.fit(x_train, y_train)
    pred = cl.predict(x_test)
    acc = accuracy_score(y_true=y_test, y_pred=pred)
    logging.info('acc with extracted dimensions: ' + str(acc))

    shrink_dim_and_plot_2d_clusters(neg_v=extracted_neg_wordvectors,
                                    pos_v=extracted_pos_wordvectors,
                                    reduction_methode=reduction_methode,
                                    bias=bias,
                                    perplexity=perplexity,
                                    learning_rate=learning_rate,
                                    normalize=normalize,
                                    extract_dim=extract_dim,
                                    truncate_by_svd=truncate_by_svd,
                                    source='word')

    ############## plot most informative dimensions ##############
    # pos_index_21 = []
    # pos_index_119 = []
    # neg_index_21 = []
    # neg_index_119 = []
    # for neg_v, pos_v in zip(neg_v, pos_v):
    #     pos_index_21.append(pos_v[21])
    #     pos_index_119.append(pos_v[119])
    #     neg_index_21.append(neg_v[21])
    #     neg_index_119.append(neg_v[119])
    #
    # negative_reduced = []
    # positive_reduced = []
    # [negative_reduced.append([v21, v119]) for v21, v119 in zip(neg_index_21, neg_index_119)]
    # [positive_reduced.append([v21, v119]) for v21, v119 in zip(pos_index_21, pos_index_119)]
    #
    # vis.plot_relevant_indexes(neg_index_21, neg_index_119, pos_index_21, pos_index_119, 'words')


def plot_sentiment_distribution(neg_v, pos_v, source=None):
    pos_index_21 = []
    pos_index_119 = []
    neg_index_21 = []
    neg_index_119 = []
    for neg_v, pos_v in zip(neg_v, pos_v):
        pos_index_21.append(pos_v[21])
        pos_index_119.append(pos_v[119])
        neg_index_21.append(neg_v[21])
        neg_index_119.append(neg_v[119])

    negative_reduced = []
    positive_reduced = []
    [negative_reduced.append([v21, v119]) for v21, v119 in zip(neg_index_21, neg_index_119)]
    [positive_reduced.append([v21, v119]) for v21, v119 in zip(pos_index_21, pos_index_119)]

    vis.plot_relevant_indexes(neg_index_21, neg_index_119, pos_index_21, pos_index_119, source)


def shuffle_and_split(neg, pos):
    logging.info('acc with reduced data')

    start_time = time.time()
    neg_labels = []
    pos_labels = []
    for _ in neg:
        neg_labels.append(c.NEGATIVE)
    for _ in pos:
        pos_labels.append(c.POSITIVE)

    logging.info("feats splitted in training and testing data - %6.2f seconds ---" % (time.time() - start_time))

    # split data into testing and training set + shuffle
    return train_test_split(
        list(neg) + list(pos), list(neg_labels) + list(pos_labels), test_size=0.25, random_state=42)

def get_tsne_model_singelton(perplexity=80, learning_rate=1000):
    global tsne_singelton_model
    if tsne_singelton_model==None:
        logging.info('create a new tsne-model')
        tsne_singelton_model = TSNE(n_components=2,
                         perplexity=perplexity,
                         verbose=2,
                         learning_rate=learning_rate)

    return tsne_singelton_model

def get_svd_model_singelton(n_components=50, random_state=0):
    global svd_singelton_model
    if svd_singelton_model==None:
        logging.info('create a new tsvd-model')
        svd_singelton_model = TruncatedSVD(n_components=50, random_state=0)

    return svd_singelton_model

def get_normalizer_singelton():
    global normalizer_singelton_model
    if normalizer_singelton_model==None:
        logging.info('create a new normalizer-model')
        normalizer_singelton_model = preprocessing.StandardScaler()

    return normalizer_singelton_model


def calc_acc(neg, pos):

    x_train, x_test, y_train, y_test = shuffle_and_split(neg, pos)

    # parameters works best for word2vec from google
    cl = LinearSVC(C=3, tol=0.1)
    cl.fit(x_train,y_train)
    pred = cl.predict(x_test)
    acc = accuracy_score(y_true=y_test,y_pred= pred)
    logging.info('acc after tsne: ' + str(acc))


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

# method to provide batch reduction to the whole review dataset.
def shrink_dim_to_2d(feats, reduction_methode='tsne', bias=None, perplexity=80, learning_rate=1000, normalize=True, extract_dim=None, truncate_by_svd=True, source='word or feat'):

    #take the first n feats, they are randomized so we can take the first 2000 - avoid memory error

    input_dimension = len(feats[0])
    logging.info('input dimensions before reduction: ' + str(input_dimension))
    if input_dimension == 2:
        # already in the right shape return in shape of 2d
        return feats

    else:

        # first reduce the dimensions to 50, then perform t-SNE or PCA
        if truncate_by_svd:
            try:
                start_time = time.time()
                svd_model = get_svd_model_singelton(n_components=50, random_state=0)
                try:
                    feats = svd_model.transform(feats)
                except:
                    feats = svd_model.fit_transform(feats)

                logging.info("dimension truncated with SVD - %6.2f seconds " % (time.time() - start_time))
            except:
                logging.info('truncating not possible, dimension < 50')

        #reduce dimension with TSNE or PCA
        if reduction_methode == 'tsne':
            # data mixed before dimension reduction
            start_time = time.time()
            # model = TSNE(n_components=goal_dimensions, perplexity=perplexity, early_exaggeration=early_exaggeration, n_iter=n_iter, learning_rate=learning_rate, random_state=0)
            model = get_tsne_model_singelton(perplexity=perplexity, learning_rate=learning_rate)

            # fit transform with the first batch, later only transform
            try:
                reduced_data = model.transform(feats)
            except:
                reduced_data = model.fit_transform(feats)

            logging.info("dimension reduced with TSNE - %6.2f seconds " % (time.time() - start_time))
            # negative and positive separately shrinked
            # neg_v_reduced, pos_v_reduced = reduce_with_TSNE(neg_v=neg_v, pos_v=pos_v, goal_dimensions=2)
        elif reduction_methode == 'pca':
            #   TODO: implement PCA at this point
            pass

        # normalize the data, fit_transform the first time, later only transform
        if normalize:
            normalizer = get_normalizer_singelton()
            try:
                reduced_data = normalizer.transform(reduced_data)
            except:
                reduced_data = normalizer.fit_transform(reduced_data)
            pass

        # return the data in the 2d shape
        return reduced_data


def shrink_dim_and_plot_2d_clusters(neg_v, pos_v, reduction_methode, bias=None, perplexity=None, learning_rate=None, normalize=True, extract_dim=None, truncate_by_svd=True, source='word or feat'):

    #take the first n feats, they are randomized so we can take the first 2000 - avoid memory error

    input_dimension = len(neg_v[0])
    logging.info('input dimensions before reduction: ' + str(input_dimension))
    if input_dimension == 2:
        calc_acc(neg_v, pos_v)
        # print 2d
        vis.plot_2d_clusters(v_neg_reduced=neg_v,
                             v_pos_reduced=pos_v,
                             filename=source + '_' + reduction_methode + '_'
                                      + 'b_' + str(bias) + '_'
                                      + 'len_' + str(len(neg_v) + len(pos_v)) + '_'
                                      + 'perpl_' + str(perplexity) + '_'
                                      + 'learn_' + str(learning_rate) + '_'
                                      + 'filter_' + str(extract_dim) + '_'
                                      + 'norm_' + str(normalize))

    else:

        # first reduce the dimensions to 50, then perform t-SNE or PCA
        if truncate_by_svd:
            try:
                start_time = time.time()

                truncated = TruncatedSVD(n_components=50, random_state=0).fit_transform(neg_v + pos_v)
                # split the truncated
                neg_v = truncated[0:int(len(truncated) / 2)]
                pos_v = truncated[int(len(truncated) / 2):]

                logging.info("dimension truncated with SVD - %6.2f seconds " % (time.time() - start_time))
            except:
                logging.info('truncating not possible, dimension < 50')

        #reduce dimension with TSNE or PCA
        if reduction_methode == 'tsne':
            # data mixed before dimension reduction
            neg_v, pos_v = vec.reduce_with_TSNE_mixed(neg_v=neg_v, pos_v=pos_v,
                                                                      goal_dimensions=2, perplexity=perplexity,
                                                                      learning_rate=learning_rate)

            # negative and positive separately shrinked
            # neg_v_reduced, pos_v_reduced = reduce_with_TSNE(neg_v=neg_v, pos_v=pos_v, goal_dimensions=2)
        elif reduction_methode == 'pca':
            neg_v, pos_v = vec.reduce_with_PCA_mixed(neg_v=neg_v, pos_v=pos_v,
                                                                     goal_dimensions=2)

        # normalize the data
        if normalize:
            scaler = preprocessing.StandardScaler().fit(neg_v + pos_v)
            neg_v = scaler.transform(neg_v)
            pos_v = scaler.transform(pos_v)


        calc_acc(neg_v,pos_v)

        # print 2d
        vis.plot_2d_clusters(v_neg_reduced=neg_v,
                             v_pos_reduced=pos_v,
                             filename=source + '_' + reduction_methode + '_'
                                      + 'b_' + str(bias) + '_'
                                      + 'len_' + str(len(neg_v) + len(pos_v)) + '_'
                                      + 'perpl_' + str(perplexity) + '_'
                                      + 'learn_' + str(learning_rate) + '_'
                                      + 'filter_' + str(extract_dim) + '_'
                                      + 'norm_' + str(normalize))



def plot_each_review_dimension(vectorized_data, bias=0.1):
    logging.info('negative vectors in vetorized[train_neg_v] : ' + str(len(vectorized_data['train_neg_v'])))
    logging.info('positive vectors in vetorized[train_pos_v] : ' + str(len(vectorized_data['train_pos_v'])))

    ############# plot each dimension to find the significant dimensions #########
    avg = []
    avg_v_neg = vec.avg_vectors(vectorized_data['train_neg_v'])
    avg_v_pos = vec.avg_vectors(vectorized_data['train_pos_v'])

    # calculate a difference vector for all averaged neg and pos vectors
    diff_v = vec.diff(avg_v_neg, avg_v_pos, bias=bias)

    # diff_v = normalize(diff_v)
    avg.append(avg_v_neg)
    avg.append(avg_v_pos)
    vis.plot_each_dim(neg_v=vectorized_data['train_neg_v'], pos_v=vectorized_data['train_pos_v'], avgs=avg, used_bias=bias,diff=diff_v, filename='feats')






def use_word2vec_with_movie_reviews():
    clf = cls.LinearSVM()

    # samples per sentiment for cluster plotting
    samples= 10000

    # tsne related params
    perplexity = 80
    # filter the most significant dimensions

    #learning_rates = np.logspace(2, 3, 5)
    learning_rates = [1000]
    # how to reduce the dimensionality of the wordvectors / document vectors
    reduction_methode = 'tsne'

    extract_dim = True
    normalize = True
    truncate_by_svd=True

    # bias for the difference of all averaged document vectors
    # how big should the difference between negative and positive feats be?
    # biases = np.array([0.1,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02, 0.01, 0.009, 0.008, 0.007,0.006])
    biases = np.array([0.09])
    accuracies = np.zeros(len(biases))
    extracted_dim = np.zeros(len(biases))

    logging.info(biases)
    logging.info(extracted_dim)
    logging.info(accuracies)

    # cache the vectorized features for faster parameter research
    import thesis.IO_Organizer as saver
    feature_filename = 'w2v_google'
    try:
        logging.info('Try to load vectorized features')
        vectorized_data_full = saver.load_features('dict_' + feature_filename)
        logging.info('Features loaded from files')
    except:
        logging.info('Feature-file not found, vectorize reviews')
        data = Data_loader().get_data()
        word2vec = vec.get_Vectorizer(vectorizer='word2vec')
        vectorized_data_full = word2vec.vectorize(data=data)
        saver.save_features(vectorized_data_full,feature_filename)

    data = Data_loader().get_data()
    word2vec = vec.get_Vectorizer(vectorizer='word2vec')
    vectorized_data_full = word2vec.vectorize(data=data)

    for learning_rate in learning_rates:
        for i, bias in enumerate(biases):
            logging.info(bias)
            # create a working copy
            vectorized_data = dict(vectorized_data_full)


            ############## plot most informative dimensions ##############
            #plot_sentiment_distribution(vectorized_data['train_neg_v'], vectorized_data['train_pos_v'], source='feats')

            # reduce the dim of our document vectors
            #vectorized_data = vec.transform_data(vectorized_data, bias=bias)

            # plotting
            plot_each_review_dimension(vectorized_data=vectorized_data, bias=bias)

            # # extract the most significant dim of our document vectors
            if extract_dim:
                vectorized_data = vec.transform_data(vectorized_data, bias=bias)


            #### testing purpose, shrinking the whole amount of data to 2d
            # we need to do it batchsized to avoid memory overflow
            batchsize = 4000
            reduced_to_2d = []
            for x in batch(vectorized_data['x_train_v'], batchsize):
                reduced_to_2d.extend(shrink_dim_to_2d(x))
            vectorized_data['x_train_v'] = reduced_to_2d
            reduced_to_2d = []

            for x in batch(vectorized_data['x_test_v'], batchsize):
                reduced_to_2d.extend(shrink_dim_to_2d(x))
            vectorized_data['x_test_v'] = reduced_to_2d
            reduced_to_2d = []

            for x in batch(vectorized_data['train_neg_v'], batchsize):
                reduced_to_2d.extend(shrink_dim_to_2d(x))
            vectorized_data['train_neg_v'] = reduced_to_2d
            reduced_to_2d = []

            for x in batch(vectorized_data['train_pos_v'], batchsize):
                reduced_to_2d.extend(shrink_dim_to_2d(x))
            vectorized_data['train_pos_v'] = reduced_to_2d
            reduced_to_2d = []

            ####


            shrink_dim_and_plot_2d_clusters(neg_v= vectorized_data['train_neg_v'],
                                                       pos_v= vectorized_data['train_pos_v'],
                                                       reduction_methode= reduction_methode,
                                                       bias= bias,
                                                       perplexity= perplexity,
                                                       learning_rate= learning_rate,
                                                       normalize= normalize,
                                                       extract_dim= extract_dim,
                                                       truncate_by_svd= truncate_by_svd,
                                                       source= 'feat')


            # select num_of_samples randomly
            # we need to define samples, or we get an memory error
            # neg_samples_v = random.sample(vectorized_data['train_neg_v'], k=samples)
            # pos_samples_v = random.sample(vectorized_data['train_pos_v'], k=samples)

            # shrink_dim_and_plot_2d_clusters(neg_v= neg_samples_v,
            #                                            pos_v= pos_samples_v,
            #                                            reduction_methode= reduction_methode,
            #                                            bias= bias,
            #                                            perplexity= perplexity,
            #                                            learning_rate= learning_rate,
            #                                            normalize= normalize,
            #                                            extract_dim= extract_dim,
            #                                            truncate_by_svd= truncate_by_svd,
            #                                            source= 'feat')


            extr_dim = len(vectorized_data['x_train_v'][0])
            extracted_dim[i] = extr_dim


            #vectorized_data = vec.delete_relevant_dimensions(vectorized_data)

            ######## linear svm ################
            cl = cls.LinearSVM()
            cl.classify(vectorized_data)
            cl.predict(vectorized_data)


            cl = LinearSVC()
            cl.fit(vectorized_data['x_train_v'], vectorized_data['y_train'])
            pred = cl.predict(vectorized_data['x_test_v'])
            acc = accuracy_score(y_true=vectorized_data['y_test'],y_pred= pred)
            logging.info('acc: ' + str(acc))
            accuracies[i] = acc
            del vectorized_data
            #
            #vis.plot_hyperplane(clf=cl, X=vectorized_data['x_train_v'], Y=vectorized_data['y_train'])


    #         ######### RandomForestClassifier #########
    #         target_names = ['negative', 'positive']
    #
    #         clf = RandomForestClassifier(n_jobs=2)
    #         clf.fit(vectorized_data['x_train_v'], vectorized_data['y_train'])
    #         prediction = clf.predict(vectorized_data['x_test_v'])
    #         logging.info(classification_report(vectorized_data['y_test'], prediction,
    #                                            target_names=target_names))
    #         ######## Logisticregression #############
    #         from sklearn.linear_model import LogisticRegression
    #         import pandas as pd
    #
    #         lr = LogisticRegression()
    #         lr.fit(vectorized_data['x_train_v'], vectorized_data['y_train'])
    #         prediction = lr.predict_proba(vectorized_data['x_test_v'])
    #
    #         logging.info('LR acc: ' + str(lr.score(vectorized_data['x_test_v'], vectorized_data['y_test'])))
    #
    #         metrics.accuracy_score(vectorized_data['y_test'], prediction)
    #
    logging.info(biases)
    logging.info(extracted_dim)
    logging.info(accuracies)


def plot_bias():


    # values gained by raising the bias threshhold
    # biases = np.array([0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002])
    # extracted_dim = np.array([2., 3., 5., 10., 22., 40., 69., 106., 165., 242., 250., 255., 257., 268., 273., 278., 286., 291.])
    # acc = np.array([0.6745, 0.69683333, 0.69625, 0.73083333, 0.77433333, 0.79225, 0.79941667, 0.81966667, 0.83083333, 0.84666667,0.84733333, 0.84858333, 0.8475, 0.85083333, 0.8515, 0.8523333, 0.85308333, 0.85591667])
    biases = np.array([0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01, 0.011,0.012,0.013])
    extracted_dim = np.array([ 255.,205.,167.,131.,108., 80., 60., 45., 28., 21., 12.,9.,6.])
    acc = np.array([0.8435, 0.83783333, 0.835, 0.82833333, 0.82116667, 0.81658333,0.80641667, 0.79791667, 0.78483333, 0.769, 0.73633333, 0.72758333, 0.6945])
    acc = [elem *100 for elem in acc]

    vis.plot_acc_for_bias(biases=biases, dimensions=extracted_dim, accs=acc)


if __name__ == "__main__":
    # testing ourpose
    #plot_bias()
    #use_word2vec_with_wordlists()
    use_word2vec_with_movie_reviews()
