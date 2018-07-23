import numpy as np
import os
import time

from matplotlib import pyplot as plt
from sklearn import svm, preprocessing
from scipy import signal
from matplotlib.colors import Normalize

PATH = './logs/'
CLASS_NAME = 'VISUALISATION'

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_plot(fig, methode, filename=''):
    dir = PATH + CLASS_NAME + '/' + methode + '/'
    ensure_dir(dir)
    i = 0
    while True:
        i += 1
        newname = '{}{:d}.png'.format(filename + '_', i)
        if os.path.exists(dir + newname):
            continue
        fig.savefig(dir + newname)
        break


# Utility function to move the midpoint of a colormap to be around
# the values of interest.

class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))




### plot top features
# example call:
# plotter.plot_coefficients(classifier_liblinear, vectorizer.get_feature_names())

def plot_coefficients(classifier, feature_names, fname, top_features=10):
    METHODE_NAME = 'plot_coefficients'
    coef = []
    try:
        coef = classifier.coef_.ravel()
    except:
        # neccessary for plotting svc(kernel=linear) we have to transform the sparse matrix into an array
        coef = classifier.coef_.toarray().ravel()

    #scaler = preprocessing.StandardScaler()
    #coef = scaler.fit_transform(coef)


    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    plt.figure()
    colors = ['blue' if c < 0 else 'red' for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(0, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
    plt.tight_layout()
    #plt.show()
    save_plot(plt.gcf(),METHODE_NAME,fname)





def plot_hyperplane(clf, X, Y):
    # fit the model
    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(-5, 5)
    yy = a * xx - (clf.intercept_[0]) / w[1]

    # plot the parallels to the separating hyperplane that pass through the
    # support vectors
    b = clf.support_vectors_[0]
    yy_down = a * xx + (b[1] - a * b[0])
    b = clf.support_vectors_[-1]
    yy_up = a * xx + (b[1] - a * b[0])

    # plot the line, the points, and the nearest vectors to the plane
    plt.plot(xx, yy, 'k-')
    plt.plot(xx, yy_down, 'k--')
    plt.plot(xx, yy_up, 'k--')

    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                s=80, facecolors='none')
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)

    plt.axis('tight')
    plt.show()



def plot_each_dim(neg_v, pos_v, avgs=None, diff=None, used_bias=None, filename='noname', example_plots = 6):
    import matplotlib.pyplot as plt
    import random
    FOLDER = 'per_dimension'

    fig = plt.figure()
    subplot_field = 411
    # 411 = 4 images, first row, first line

    # define a subplot for the neg examples
    plt.subplot(subplot_field)
    # plot one line per neg examples
    for vector in random.sample(neg_v, k=example_plots):
        plt.plot(vector, linewidth=.40)
    plt.title(str(example_plots) + ' neg examples from ' + str(len(neg_v)) + ' ' + filename)
    subplot_field += 1


    # define a subplot for the pos examples
    plt.subplot(subplot_field)
    # plot one line per pos example
    for vector in random.sample(pos_v, k=example_plots):
        plt.plot(vector, linewidth=.40)
    plt.title(str(example_plots) + ' pos examples from ' + str(len(pos_v)) + ' ' + filename)
    subplot_field += 1



    # define a subplot for the avg lines
    plt.subplot(subplot_field)
    for vector, label in zip(avgs, ['neg_avg', 'pos_avg']):
        plt.plot(vector, linewidth=.40, label=label)
    plt.title('avg of ' + str(len(neg_v)) +  ' ' + filename)
    subplot_field += 1

    plt.legend(scatterpoints=1,ncol=3,fontsize=8)

    # define a subplot for the diff

    diff = np.array(diff)
    plt.subplot(subplot_field)
    plt.plot(diff, linewidth=.40)
    # calculate the argmax
    idx = signal.argrelextrema(diff, np.greater)
    # marc the max values
    plt.plot(idx[0], diff[idx], 'ro', label=idx[0], ms=2.0)
    #plt.legend(scatterpoints=1, ncol=3, fontsize=8)
    plt.xlabel('dimensions')

    plt.title('difference ' + str(len(neg_v)) + ' used bias: ' + str(used_bias))
    subplot_field += 1


    plt.tight_layout()
    save_plot(plt.gcf(), FOLDER, filename)



def plot_2d_clusters(v_neg_reduced, v_pos_reduced, filename):
    METHODE_NAME= '2d_cluster'

    # print 2d
    plt.figure(figsize=(6.4,4.8))
    neg = plt.scatter(*zip(*v_neg_reduced), s=1, marker='o', color='r', label='neg')
    pos = plt.scatter(*zip(*v_pos_reduced), s=1, marker='o', color='b', label='pos')
    plt.legend((neg, pos),
               ('negative', 'positive'),
               scatterpoints=1,
               loc='lower right',
               ncol=3,
               fontsize=8)

    plt.title(filename)
    fig = plt.gcf()

    # store image to file
    # always a new name for the image, avoid rewriting

    save_plot(fig, METHODE_NAME, filename)



def plot_relevant_indexes(neg_21, neg_119, pos_21, pos_119, filename=''):
    METHODE_NAME = 'sentiment_distribution'
    col = 4
    row = 4
    fig = plt.figure(figsize=(15,8))
    fig.suptitle('pos feats: ' + str(len(neg_21)) + 'neg feats: ' + str(len(pos_21)))

    neg_21_t = neg_21[:]
    neg_119_t = neg_119[:]
    pos_21_t = pos_21[:]
    pos_119_t = pos_119[:]


    import numpy as np
    import scipy.stats as stats


    ############################
    # distribution of v[21] normalized
    ############################
    plt.subplot2grid((col,row), (0,0), colspan=2)
    neg_21.sort()
    hmean = np.mean(neg_21)
    hstd = np.std(neg_21)
    pdf = stats.norm.pdf(neg_21, hmean, hstd)
    plt.plot(neg_21, pdf, label='neg[21]')

    pos_21.sort()
    hmean = np.mean(pos_21)
    hstd = np.std(pos_21)
    pdf = stats.norm.pdf(pos_21, hmean, hstd)
    plt.plot(pos_21, pdf, label='pos[21]')

    plt.ylabel('norm freq')
    plt.xlabel('value of v[21]')
    plt.title('value distibution of v[21]')
    plt.tight_layout()
    plt.legend(scatterpoints=1,
               loc='upper right',
               ncol=3,
               fontsize=8)

    ##########################
    # distibution of v[119] normalized
    ##########################

    plt.subplot2grid((col,row), (1,0), colspan=2)
    neg_119.sort()
    hmean = np.mean(neg_119)
    hstd = np.std(neg_119)
    pdf = stats.norm.pdf(neg_119, hmean, hstd)
    plt.plot(neg_119, pdf, label='neg[119]')

    pos_119.sort()
    hmean = np.mean(pos_119)
    hstd = np.std(pos_119)
    pdf = stats.norm.pdf(pos_119, hmean, hstd)
    plt.plot(pos_119, pdf, label='pos[119]')

    plt.ylabel('norm freq')
    plt.xlabel('value of v[119]')
    plt.title('value distibution of v[119]')
    plt.tight_layout()
    plt.legend(scatterpoints=1,
               loc='upper right',
               ncol=3,
               fontsize=8)

    ##########################
    # frequency raw v[21]
    ##########################


    plt.subplot2grid((col, row), (2, 0), colspan=2)
    # number of steps
    bins = 50

    plt.hist(neg_21, bins=bins, histtype='step', label='neg[21]')
    plt.hist(pos_21, bins=bins, histtype='step', label='pos[21]')


    plt.ylabel('frequency of v[21]')
    plt.xlabel('value of v[n]')
    plt.title('value frequency of v[n]')
    plt.xlim(-1,1)
    plt.tight_layout()
    plt.legend(scatterpoints=1,
               loc='upper left',
               ncol=3,
               fontsize=8)


    ##########################
    # frequency raw v[119]
    ##########################

    plt.subplot2grid((col, row), (3, 0), colspan=2)
    # number of steps
    bins = 50
    plt.hist(neg_119, bins=bins, histtype='step', label='neg[119]')
    plt.hist(pos_119, bins=bins, histtype='step', label='pos[119]')

    plt.ylabel('frequency of v[119]')
    plt.xlabel('value of v[n]')
    plt.title('value frequency of v[n]')
    plt.xlim(-1,1)
    plt.tight_layout()
    plt.legend(scatterpoints=1,
               loc='upper left',
               ncol=3,
               fontsize=8)


    ##########################
    # distribution x = v[119], y = v[21]
    ##########################

    plt.subplot2grid((col,row), (0,2), colspan=2, rowspan=2)
    neg = plt.scatter(neg_21_t, neg_119_t, s=1)
    pos = plt.scatter(pos_21_t, pos_119_t, s=1)
    plt.ylabel('vec[21]')
    plt.xlabel('vec[119]')
    plt.title('distibution of v[21] and v[119]')
    plt.tight_layout()
    plt.legend((neg, pos),
               ('neg', 'pos'),
               scatterpoints=1,
               loc='lower right',
               ncol=3,
               fontsize=8)
    fig = plt.gcf()
    save_plot(fig, METHODE_NAME, filename=filename)
    #plt.show()


def plot_heatmap(scores, gamma_range, C_range, filename='no_filename_given'):
    METHODE_NAME = 'heatmap'

    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    # np.set_printoptions(precision=4)
    plt.figure(figsize=(8, 6))
    # plt.gca().get_xaxis().set_major_formatter(mtick.FormatStrFormatter('%.4f'))
    # plt.gca().get_yaxis().set_major_formatter(mtick.FormatStrFormatter('%.4f'))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
               norm=MidpointNormalize(vmin=0.6, midpoint=0.83))
    plt.xlabel('SVM parameter tol')
    plt.ylabel('SVM parameter C')
    plt.colorbar()

    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title('Validation accuracy')
    plt.tight_layout()
    fig = plt.gcf()
    save_plot(fig, METHODE_NAME, filename=filename)


def plot_acc_for_bias(biases, dimensions, accs, filename='movie_review'):
    METHODE_NAME = 'acc_by_dim'

    fig = plt.figure()
    plt.title('Accuracy per extracted dimensions')

    plt.plot(dimensions, accs, label='dim')
    plt.scatter(dimensions, accs, s=5, color='black')
    #plt.plot(biases, accs, label='acc')
    #plt.scatter(biases, accs, s=5, color='black')
    plt.xlabel('Extracted dimensions')
    plt.ylabel('Accuracy in %')
    #plt.legend(scatterpoints=1,ncol=3,fontsize=8)

    plt.tight_layout()
    fig = plt.gcf()
    save_plot(fig, METHODE_NAME, filename=filename)
    #plt.show()


def learning_curves():
    print(__doc__)

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    from sklearn.datasets import load_digits
    from sklearn.model_selection import learning_curve
    from sklearn.model_selection import ShuffleSplit

    def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                            n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
        """
        Generate a simple plot of the test and training learning curve.

        Parameters
        ----------
        estimator : object type that implements the "fit" and "predict" methods
            An object of that type which is cloned for each validation.

        title : string
            Title for the chart.

        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples) or (n_samples, n_features), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        ylim : tuple, shape (ymin, ymax), optional
            Defines minimum and maximum yvalues plotted.

        cv : int, cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:
              - None, to use the default 3-fold cross-validation,
              - integer, to specify the number of folds.
              - An object to be used as a cross-validation generator.
              - An iterable yielding train/test splits.

            For integer/None inputs, if ``y`` is binary or multiclass,
            :class:`StratifiedKFold` used. If the estimator is not a classifier
            or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

            Refer :ref:`User Guide <cross_validation>` for the various
            cross-validators that can be used here.

        n_jobs : integer, optional
            Number of jobs to run in parallel (default 1).
        """
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")

        plt.legend(loc="best")
        return plt

    def plot_learning(vectorized_data):
        X = vectorized_data['x_train_v'].toarray()
        y = vectorized_data['y_train']

        # title = "Learning Curves (Naive Bayes)"
        # # Cross validation with 100 iterations to get smoother mean test and train
        # # score curves, each time with 20% data randomly selected as a validation set.
        # cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
        #
        # estimator = GaussianNB()
        # plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

        title = "Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
        # SVC is more expensive so we do a lower number of CV iterations:
        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
        estimator = svm.LinearSVC()
        # estimator = SVC(gamma=0.001)
        plot_learning_curve(estimator, title, X, y, (0.7, 1.01), cv=cv, n_jobs=4)

        plt.show()