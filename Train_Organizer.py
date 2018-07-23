# --- Own Libs --- #
import data.Data_Organizer as data_org
from svm.SVM_based_Classifier import run_svm_based_classifier_new
from svm.SVM_based_Classifier import run_svm_based_classifier
from dictionary.Lexicon_based_Classifier import run_lexicon_based_classifier
from naive_bayes.Naive_Bayes_based_Classifier import run_naive_based_classifier


def svm_based_evaluation(negfeats, posfeats):
    #run_svm_based_classifier(negfeats, posfeats)
    run_svm_based_classifier_new(negfeats, posfeats)

def lexicon_based_evaluation(negfeats, posfeats):
    run_lexicon_based_classifier(negfeats, posfeats)


def naive_bayes_based_evaluation(negfeats, posfeats):
    run_naive_based_classifier(negfeats,posfeats)


# naive bayes is the standard evaluation method
def evaluate_classifier(featx, evaluation_method= naive_bayes_based_evaluation):

    evaluation_method = svm_based_evaluation
    #evaluation_method = lexicon_based_evaluation

    # load the training- and testdata
    negfeats, posfeats = data_org.load_data_for_svm(featx)

    # evaluate (train and test)
    evaluation_method(negfeats, posfeats)



evaluate_classifier(data_org.stopword_filtered_word_feats)