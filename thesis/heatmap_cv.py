# import modules & set up logging
from sklearn.manifold import TSNE
from sklearn.svm import LinearSVC

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

import thesis.Data as d
import thesis.Vectorizer as vec
import thesis.my_logger
import thesis.Visualization as plotter


# tfidf
# data = d.Data_loader().get_data()
# tfidf_vec = vec.get_Vectorizer('tfidf')
# vectorized_data = tfidf_vec.vectorize(data=data)

# word2vec
data = d.Data_loader().get_data()
word2vec_vec = vec.get_Vectorizer('word2vec')
vectorized_data = word2vec_vec.vectorize(data=data)



X = vectorized_data['x_train_v']
y = vectorized_data['y_train']

print('grid')
C_range = np.logspace(-2, 2, 5)
gamma_range = np.logspace(-4, 2, 5)

param_grid = dict(tol=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=42)
grid = GridSearchCV(LinearSVC(), param_grid=param_grid, cv=cv, verbose=1)
grid.fit(X, y)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))
print('grid done')

scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),
                                                     len(gamma_range))

# Draw heatmap of the validation accuracy as a function of gamma and C
#
# The score are encoded as colors with the hot colormap which varies from dark
# red to bright yellow. As the most interesting scores are all located in the
# 0.82 to 0.85 range we use a custom normalizer to set the mid-point to 0.82 so
# as to make it easier to visualize the small variations of score values in the
# interesting range while not brutally collapsing all the low score values to
# the same color.
plotter.plot_heatmap(scores=scores, gamma_range=gamma_range, C_range=C_range, filename='linear_SVM')
