from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier
#from lightgbm import LGBMClassifier

#except for Neural Network for now
CLASSIFIERS = [
    ('Random Forest', RandomForestClassifier(n_estimators=100)),
    ('Logistic Regression', LogisticRegression()),
    ('Support Vector Machine', SVC(gamma='auto', probability=True)),
    ('k-Nearest Neighbors', KNeighborsClassifier()),
    ('Decision Tree', DecisionTreeClassifier()),
    ('Naive Bayes', GaussianNB()),
    ('Extreme Gradient Boosting', XGBClassifier()),
    #('LASSO', Lasso()),
    #('Stochastic Gradient Descent', SGDClassifier()),
    ('Linear Discriminant Analysis', LinearDiscriminantAnalysis())
    #('Gradient Boosted Machine', LGBMClassifier(n_estimators=90, silent=False, random_state =94, max_depth=5,num_leaves=31,objective='binary',metrics ='auc'))
]

REGRESSORS = [
    ('Random Forest', RandomForestRegressor()),
    ('Linear Regression', LinearRegression()),
    ('k-Nearest Neighbors', KNeighborsRegressor()),
    ('Decision Tree', DecisionTreeRegressor())
    #('Gradient Boosted Machine', LGBMClassifier(n_estimators=90, silent=False, random_state =94, max_depth=5,num_leaves=31,objective='binary',metrics ='auc'))
]


Cs = [0.001, 0.01]
gammas = [0.001, 0.01]

PARAMS = [
    {'C': Cs, 'gamma' : gammas}
]

