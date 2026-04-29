"""Tune model hyperparameters using Optuna."""

from sklearn import neighbors, metrics, tree, ensemble
from time import time
from random import choice, randint, uniform

class ObjectiveKnn:
	def __init__(self, X_train, X_val, y_train, y_val):
		self.X_train = X_train
		self.X_val = X_val
		self.y_train = y_train
		self.y_val = y_val

	def __call__(self, trial):
		n_neighbors = trial.suggest_int('n_neighbors', 1, 159)
		metric = trial.suggest_categorical('metric', ['euclidean', 'cityblock'])
		weights = trial.suggest_categorical('weights', ['uniform', 'distance'])

		classifier = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, weights=weights)
		classifier.fit(self.X_train, self.y_train)

		y_pred = classifier.predict(self.X_val)
		return metrics.f1_score(self.y_val, y_pred, average='weighted')

class ObjectiveDt:
	def __init__(self, X_train, X_val, y_train, y_val):
		self.X_train = X_train
		self.X_val = X_val
		self.y_train = y_train
		self.y_val = y_val

	def __call__(self, trial):
		criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
		ccp_alpha = trial.suggest_float('ccp_alpha', 0.0, 0.1)
		max_depth = trial.suggest_int('max_depth', 1, 21)
		if max_depth == 21:
			max_depth = None

		classifier = tree.DecisionTreeClassifier(criterion=criterion, ccp_alpha=ccp_alpha, max_depth=max_depth)
		classifier.fit(self.X_train, self.y_train)

		y_pred = classifier.predict(self.X_val)
		return metrics.f1_score(self.y_val, y_pred, average='weighted')

class ObjectiveRf:
	def __init__(self, X_train, X_val, y_train, y_val):
		self.X_train = X_train
		self.X_val = X_val
		self.y_train = y_train
		self.y_val = y_val

	def __call__(self, trial):
		n_estimators = trial.suggest_int('n_estimators', 60, 180)
		min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
		max_depth = trial.suggest_int('max_depth', 5, 21)
		if max_depth == 21:
			max_depth = None
		max_features = trial.suggest_int('max_features', 2, self.X_train.shape[1])
		bootstrap = trial.suggest_categorical('bootstrap', [True, False])

		classifier = ensemble.RandomForestClassifier(
			n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_features=max_features, bootstrap=bootstrap
		)
		classifier.fit(self.X_train, self.y_train)

		y_pred = classifier.predict(self.X_val)
		return metrics.f1_score(self.y_val, y_pred, average='weighted')

def random_search_knn(X_train, X_val, y_train, y_val):
	best_f1_score = 0

	start_time = time()
	while time() - start_time < 300:
		n_neighbors = randint(1, 159)
		metric = choice(['euclidean', 'cityblock'])
		weights = choice(['uniform', 'distance'])

		classifier = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, weights=weights)
		classifier.fit(X_train, y_train)
		y_pred = classifier.predict(X_val)
		f1 = metrics.f1_score(y_val, y_pred, average='weighted')
		if f1 > best_f1_score:
			best_f1_score = f1
			best_n_neighbors = n_neighbors
			best_metric = metric
			best_weights = weights
	return best_n_neighbors, best_metric, best_weights

def random_search_dt(X_train, X_val, y_train, y_val):
	best_f1_score = 0

	start_time = time()
	while time() - start_time < 300:
		criterion = choice(['gini', 'entropy'])
		ccp_alpha = uniform(0.0, 0.1)
		max_depth = randint(1, 21)
		if max_depth == 21:
			max_depth = None

		classifier = tree.DecisionTreeClassifier(criterion=criterion, ccp_alpha=ccp_alpha, max_depth=max_depth)
		classifier.fit(X_train, y_train)
		y_pred = classifier.predict(X_val)
		f1 = metrics.f1_score(y_val, y_pred, average='weighted')
		if f1 > best_f1_score:
			best_f1_score = f1
			best_criterion = criterion
			best_ccp_alpha = ccp_alpha
			best_max_depth = max_depth
	return best_criterion, best_ccp_alpha, best_max_depth

def random_search_rf(X_train, X_val, y_train, y_val):
	best_f1_score = 0

	start_time = time()
	while time() - start_time < 300:
		n_estimators = randint(60, 180)
		min_samples_leaf = randint(1, 10)
		max_depth = randint(5, 21)
		if max_depth == 21:
			max_depth = None
		max_features = randint(2, X_train.shape[1])
		bootstrap = choice([True, False])

		classifier = ensemble.RandomForestClassifier(
			n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_features=max_features, bootstrap=bootstrap
		)
		classifier.fit(X_train, y_train)
		y_pred = classifier.predict(X_val)
		f1 = metrics.f1_score(y_val, y_pred, average='weighted')
		if f1 > best_f1_score:
			best_f1_score = f1
			best_n_estimators = n_estimators
			best_min_samples_leaf = min_samples_leaf
			best_max_depth = max_depth
			best_max_features = max_features
			best_bootstrap = bootstrap
	return best_n_estimators, best_min_samples_leaf, best_max_depth, best_max_features, best_bootstrap