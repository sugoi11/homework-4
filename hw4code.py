import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Finds the best split using the Gini criterion.
    """

    # Sort feature values and targets
    sorted_idx = np.argsort(feature_vector)
    feature_vector = feature_vector[sorted_idx]
    target_vector = target_vector[sorted_idx]

    # Possible thresholds: average of neighboring feature values
    thresholds = (feature_vector[:-1] + feature_vector[1:]) / 2

    # Remove duplicate thresholds
    valid = feature_vector[:-1] != feature_vector[1:]
    thresholds = thresholds[valid]

    if len(thresholds) == 0:
        return np.array([]), np.array([]), None, None

    n = len(target_vector)
    classes = np.unique(target_vector)
    n_classes = len(classes)

    ginis = []

    for threshold in thresholds:

        left_mask = feature_vector < threshold
        right_mask = ~left_mask

        y_left = target_vector[left_mask]
        y_right = target_vector[right_mask]

        if len(y_left) == 0 or len(y_right) == 0:
            continue

        # Left Gini
        left_probs = np.array([
            np.sum(y_left == c) / len(y_left)
            for c in classes
        ])
        gini_left = 1 - np.sum(left_probs ** 2)

        # Right Gini
        right_probs = np.array([
            np.sum(y_right == c) / len(y_right)
            for c in classes
        ])
        gini_right = 1 - np.sum(right_probs ** 2)

        # Weighted Gini gain
        gini = -(
            (len(y_left) / n) * gini_left +
            (len(y_right) / n) * gini_right
        )

        ginis.append(gini)

    ginis = np.array(ginis)

    # Best split
    best_idx = np.argmax(ginis)

    threshold_best = thresholds[best_idx]
    gini_best = ginis[best_idx]

    return thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    """
    Simple Decision Tree Classifier
    """

    def __init__(self, feature_types,
                 max_depth=None,
                 min_samples_split=None,
                 min_samples_leaf=None):

        if np.any(list(map(lambda x:
                           x != "real" and x != "categorical",
                           feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):

        # Stop conditions
        if len(np.unique(sub_y)) == 1:
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        if self._max_depth is not None and depth >= self._max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if self._min_samples_split is not None:
            if len(sub_y) < self._min_samples_split:
                node["type"] = "terminal"
                node["class"] = Counter(sub_y).most_common(1)[0][0]
                return

        feature_best = None
        threshold_best = None
        gini_best = None
        split_best = None

        for feature in range(sub_X.shape[1]):

            feature_type = self._feature_types[feature]

            # REAL FEATURES
            if feature_type == "real":

                feature_vector = sub_X[:, feature].astype(float)

            # CATEGORICAL FEATURES
            elif feature_type == "categorical":

                counts = Counter(sub_X[:, feature])

                ratios = {}

                for category in counts:
                    ratios[category] = counts[category]

                sorted_categories = sorted(ratios.keys())

                categories_map = {
                    cat: i
                    for i, cat in enumerate(sorted_categories)
                }

                feature_vector = np.array([
                    categories_map[x]
                    for x in sub_X[:, feature]
                ])

            else:
                raise ValueError

            # Skip constant feature
            if len(np.unique(feature_vector)) == 1:
                continue

            _, _, threshold, gini = find_best_split(
                feature_vector,
                sub_y
            )

            if threshold is None:
                continue

            if gini_best is None or gini > gini_best:

                feature_best = feature
                gini_best = gini

                split = feature_vector < threshold
                split_best = split

                if feature_type == "real":
                    threshold_best = threshold

                elif feature_type == "categorical":

                    threshold_best = [
                        category
                        for category, value
                        in categories_map.items()
                        if value < threshold
                    ]

        # No split found
        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best

        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best

        else:
            node["categories_split"] = threshold_best

        node["left_child"] = {}
        node["right_child"] = {}

        self._fit_node(
            sub_X[split_best],
            sub_y[split_best],
            node["left_child"],
            depth + 1
        )

        self._fit_node(
            sub_X[~split_best],
            sub_y[~split_best],
            node["right_child"],
            depth + 1
        )

    def _predict_node(self, x, node):

        # Terminal node
        if node["type"] == "terminal":
            return node["class"]

        feature = node["feature_split"]

        # Real feature
        if self._feature_types[feature] == "real":

            if x[feature] < node["threshold"]:
                return self._predict_node(
                    x,
                    node["left_child"]
                )
            else:
                return self._predict_node(
                    x,
                    node["right_child"]
                )

        # Categorical feature
        else:

            if x[feature] in node["categories_split"]:
                return self._predict_node(
                    x,
                    node["left_child"]
                )
            else:
                return self._predict_node(
                    x,
                    node["right_child"]
                )

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):

        predicted = []

        for x in X:
            predicted.append(
                self._predict_node(x, self._tree)
            )

        return np.array(predicted)