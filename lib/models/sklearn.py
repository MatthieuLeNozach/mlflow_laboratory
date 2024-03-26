################### -------------- ###################
################### CLASSIFICATION ###################
################### -------------- ###################


from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import (
    LogisticRegression,
    PassiveAggressiveClassifier,
    Perceptron,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.naive_bayes import (
    BernoulliNB,
    CategoricalNB,
    ComplementNB,
    GaussianNB,
    MultinomialNB,
)
from sklearn.neighbors import (
    KNeighborsClassifier,
    NearestCentroid,
    RadiusNeighborsClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import (
    LinearSVC,
    NuSVC,
    SVC,
)
from sklearn.tree import (
    DecisionTreeClassifier,
    ExtraTreeClassifier,
)
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)

SKLEARN_CLSSIFIERS = {
    'AdaBoostClassifier': AdaBoostClassifier,
    'BaggingClassifier': BaggingClassifier,
    'BernoulliNB': BernoulliNB,
    'CategoricalNB': CategoricalNB,
    'ComplementNB': ComplementNB,
    'DecisionTreeClassifier': DecisionTreeClassifier,
    'ExtraTreeClassifier': ExtraTreeClassifier,
    'ExtraTreesClassifier': ExtraTreesClassifier,
    'GaussianNB': GaussianNB,
    'GradientBoostingClassifier': GradientBoostingClassifier,
    'KNeighborsClassifier': KNeighborsClassifier,
    'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis,
    'LinearSVC': LinearSVC,
    'LogisticRegression': LogisticRegression,
    'MLPClassifier': MLPClassifier,
    'MultinomialNB': MultinomialNB,
    'NearestCentroid': NearestCentroid,
    'NuSVC': NuSVC,
    'PassiveAggressiveClassifier': PassiveAggressiveClassifier,
    'Perceptron': Perceptron,
    'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis,
    'RadiusNeighborsClassifier': RadiusNeighborsClassifier,
    'RandomForestClassifier': RandomForestClassifier,
    'RidgeClassifier': RidgeClassifier,
    'SGDClassifier': SGDClassifier,
    'StackingClassifier': StackingClassifier,
    'SVC': SVC,
    'VotingClassifier': VotingClassifier,
}


################### ---------- ###################
################### REGRESSION ###################
################### ---------- ###################



from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
    VotingRegressor,
)
from sklearn.linear_model import (
    ARDRegression,
    BayesianRidge,
    ElasticNet,
    HuberRegressor,
    Lars,
    Lasso,
    LassoLars,
    LinearRegression,
    PassiveAggressiveRegressor,
    RANSACRegressor,
    Ridge,
    SGDRegressor,
    TheilSenRegressor,
)
from sklearn.neighbors import (
    KNeighborsRegressor,
    RadiusNeighborsRegressor,
)
from sklearn.neural_network import MLPRegressor
from sklearn.svm import (
    LinearSVR,
    NuSVR,
    SVR,
)
from sklearn.tree import (
    DecisionTreeRegressor,
    ExtraTreeRegressor,
)
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.isotonic import IsotonicRegression

SKLEARN_REGRESSORS = {
    'AdaBoostRegressor': AdaBoostRegressor,
    'ARDRegression': ARDRegression,
    'BaggingRegressor': BaggingRegressor,
    'BayesianRidge': BayesianRidge,
    'DecisionTreeRegressor': DecisionTreeRegressor,
    'ElasticNet': ElasticNet,
    'ExtraTreeRegressor': ExtraTreeRegressor,
    'ExtraTreesRegressor': ExtraTreesRegressor,
    'GaussianProcessRegressor': GaussianProcessRegressor,
    'GradientBoostingRegressor': GradientBoostingRegressor,
    'HuberRegressor': HuberRegressor,
    'IsotonicRegression': IsotonicRegression,
    'KernelRidge': KernelRidge,
    'KNeighborsRegressor': KNeighborsRegressor,
    'Lars': Lars,
    'Lasso': Lasso,
    'LassoLars': LassoLars,
    'LinearRegression': LinearRegression,
    'LinearSVR': LinearSVR,
    'MLPRegressor': MLPRegressor,
    'NuSVR': NuSVR,
    'PassiveAggressiveRegressor': PassiveAggressiveRegressor,
    'PLSRegression': PLSRegression,
    'RANSACRegressor': RANSACRegressor,
    'RadiusNeighborsRegressor': RadiusNeighborsRegressor,
    'RandomForestRegressor': RandomForestRegressor,
    'Ridge': Ridge,
    'SGDRegressor': SGDRegressor,
    'StackingRegressor': StackingRegressor,
    'SVR': SVR,
    'TheilSenRegressor': TheilSenRegressor,
    'VotingRegressor': VotingRegressor,
}