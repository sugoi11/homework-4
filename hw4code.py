import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Finds the best split using Gini impurity.

    Returns:
    thresholds      - all valid thresholds
    ginis           - gini gains for each threshold
    threshold_best  - best threshold
    gini_best       - best gini gain
    """

    feature_vector = np.asarray(feature_vector)
    target_vector = np.asarray(target_vector)

    # sort by feature
    order = np.argsort(feature_vector)
    feature_sorted = feature_vector[order]
    target_sorted = target_vector[order]

    # valid split positions
    diff = feature_sorted[1:] != feature_sorted[:-1]
    valid_idx = np.where(diff)[0]

    if len(valid_idx) == 0:
        return np.array([]), np.array([]), None, None

    # thresholds = averages of neighbors
    thresholds = (
        feature_sorted[valid_idx] +
        feature_sorted[valid_idx + 1]
    ) / 2

    classes = np.unique(target_vector)
    n_classes = len(classes)
    n = len(target_vector)

    # one-hot encoding
    y_encoded = np.zeros((n, n_classes))

    for i, cls in enumerate(classes):
        y_encoded[:, i] = (target_sorted == cls)

    # cumulative class counts
    left_counts = np.cumsum(y_encoded, axis=0)
    total_counts = left_counts[-1]

    left_n = np.arange(1, n)
    right_n = n - left_n

    left_counts = left_counts[:-1]
    right_counts = total_counts - left_counts

    # only valid splits
    left_counts = left_counts[valid_idx]
    right_counts = right_counts[valid_idx]

    left_n = left_n[valid_idx]
    right_n = right_n[valid_idx]

    # probabilities
    left_p = left_counts / left_n[:, None]
    right_p = right_counts / right_n[:, None]

    # gini impurities
    gini_left = 1 - np.sum(left_p ** 2, axis=1)
    gini_right = 1 - np.sum(right_p ** 2, axis=1)

    # weighted impurity
    ginis = -(
        left_n / n * gini_left +
        right_n / n * gini_right
    )

    best_idx = np.argmax(ginis)

    threshold_best = thresholds[best_idx]
    gini_best = ginis[best_idx]

    return thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    """
    Simple Decision Tree classifier.
    Supports:
    - real and categorical features
    - binary and multiclass classification
    """

    def __init__(
        self,
        feature_types,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1
    ):

        if np.any([
            ft not in ["real", "categorical"]
            for ft in feature_types
        ]):
            raise ValueError("Unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):

        # stop if all labels equal
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        # max depth
        if self._max_depth is not None and depth >= self._max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        # min samples split
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

            # REAL FEATURE
            if feature_type == "real":
                feature_vector = sub_X[:, feature]

            # CATEGORICAL FEATURE
            elif feature_type == "categorical":

                counts = Counter(sub_X[:, feature])

                # multiclass-safe ordering
                ratios = {}

                for category in counts:
                    mask = sub_X[:, feature] == category

                    # dominant class frequency
                    cls_counts = Counter(sub_y[mask])

                    dominant_ratio = max(cls_counts.values()) / counts[category]

                    ratios[category] = dominant_ratio

                sorted_categories = sorted(
                    ratios,
                    key=ratios.get
                )

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

            if len(np.unique(feature_vector)) < 2:
                continue

            thresholds, ginis, threshold, gini = find_best_split(
                feature_vector,
                sub_y
            )

            if threshold is None:
                continue

            split = feature_vector < threshold

            # min samples leaf
            if (
                split.sum() < self._min_samples_leaf or
                (~split).sum() < self._min_samples_leaf
            ):
                continue

            if (
                gini_best is None or
                gini > gini_best or
                (
                    np.isclose(gini, gini_best) and
                    threshold < threshold_best
                )
            ):

                feature_best = feature
                threshold_best = threshold
                gini_best = gini
                split_best = split

                if feature_type == "categorical":
                    categories_split = [
                        cat
                        for cat, value in categories_map.items()
                        if value < threshold
                    ]

        # no valid split
        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best

        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        else:
            node["categories_split"] = categories_split

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

        if node["type"] == "terminal":
            return node["class"]

        feature = node["feature_split"]

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

        predictions = [
            self._predict_node(x, self._tree)
            for x in X
        ]

        return np.array(predictions)
