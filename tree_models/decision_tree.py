
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class DecisionTree(object):
    """Decision Tree.

    Implements the base class from which `ClassificationTree` and
    `RegressionTree` are built upon. All the functionality is defined in the
    base class `DecisionTree` except for evaluating a split and checking if a
    Leaf has been reached.

    Parameters
    ----------
    max_depth : int or None
        The maximum depth of the tree.
    """
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        """Fit a decision tree model to the data.

        Parameters
        ----------
        X : ndarray
            A matrix with the training data.
        y : ndarray
            A column vector with the true target values.

        Returns
        self
        """
        self.root = self.partition(X, y)
        return self

    def predict(self, X, y=None):
        """Predict the target for a data row `X`.

        Parameters
        ----------
        X : ndarray
            A data row of shape (n_features,).

        Returns
        -------
        pred : float or int
            Prediction for the data row `X`.
        """
        pred = np.zeros(shape=(X.shape[0], 1))
        for i in range(X.shape[0]):
            pred[i] = self.traverse(X[i, :], self.root)
        return pred

    def traverse(self, X, node):
        """Traverse the tree.

        Parameter
        ---------
        X : ndarray
            A single row of data.
        node: TreeNode instance
            The node should be the root of the tree.

        Returns
        -------
        pred : float
            The prediction for row `X`.
        """
        if isinstance(node, Leaf):
            return node.pred
        if X[node.feature] <= node.split_val:
            return self.traverse(X, node.left_child)
        else:
            return self.traverse(X, node.right_child)

    def partition(self, X, y, depth=0):
        """Partition the data.

        Parameters
        ----------
        X : ndarray
            A matrix with the data.
        y : ndarray
            A column vector with the true target values.
        depth : int
            The maximum depth of the tree to grow.

        Returns
        -------
        node : TreeNode instance
            A node which represents the root of the fitted tree.
        """
        check = self.check_partition(y, depth)
        if isinstance(check, Leaf):
            return check

        feature, split_val = self.find_feature(X, y)
        X_left = X[X[:, feature] <= split_val]
        X_right = X[X[:, feature] > split_val]
        y_left = y[X[:, feature] <= split_val]
        y_right = y[X[:, feature] > split_val]

        node = TreeNode(feature, split_val, self.partition(X_left, y_left, depth+1),
                        self.partition(X_right, y_right, depth+1))
        return node

    def find_feature(self, X, y):
        """Find a feature to split.

        Parameters
        ----------
        X : ndarray
        y : ndarray

        Returns
        -------
        best_feature : int
            The index of the best feature to split.
        best_split : float
            The best value to split at.
        """
        cost_low = np.inf
        best_feature = None
        best_split = None
        for i in range(X.shape[1]):
            cost, split_val = self.find_split(X[:, i], y)
            if cost < cost_low:
                best_feature = i
                best_split = split_val
                cost_low = cost
        return best_feature, best_split

    def find_split(self, feature, y):
        """Find best split.

        Find the best split given the values for the provided feature.

        Parameters
        ----------
        feature : ndarray
            An column vector with the values for one feature. Should be of
            shape (m,).
        y : ndarray
            A column vector with the true labels for the column vector
            `feature`. Should be of shape (m,).

        Returns
        -------
        cost_low : float
            The lowest cost that can be achieved by splitting the feature at
            the value `best_split`.
        best_split : float
            The value in `feature` which is the best split, i.e. the split that
            minimize the cost.
        """
        idx_sorted = np.argsort(feature)
        feature = feature[idx_sorted]
        y = y[idx_sorted]
        cost_low = np.inf
        best_split = None
        for v in np.unique(feature)[:-1]:
            r1 = y[feature <= v]
            r2 = y[feature > v]
            cost = self.eval_split(r1, r2)
            if cost < cost_low:
                cost_low = cost
                next_val = feature[feature > v][0]
                best_split = (v + next_val) / 2
        return cost_low, best_split

    def eval_split(self, r1, r2):
        raise NotImplementedError

    def check_partition(self, y, depth):
        raise NotImplementedError


class ClassificationTree(DecisionTree):
    """Classification tree.

    Implements decision tree which can be used for classification. It is built
    on the class `DecisionTree` with the only addition is to compute the cost
    function when deciding splits.

    Parameters
    ----------
    impurity_measure : {'gini', 'entropy'}
        Defines which impurity measure which should be used when fitting the
        tree.
    max_depth : int or None
        The maximum depth of the tree.
    """
    def __init__(self, impurity_measure='gini', max_depth=None):
        self.impurity_measure = impurity_measure
        super().__init__(max_depth=max_depth)

    def check_partition(self, y, depth):
        """Check if `y` only contains one class or if the maximum depth is
        reached.

        Parameters
        ----------
        y : ndarray
            An array with class labels.
        depth : int
            The maximum depth of the tree.

        Returns
        -------
        check : Leaf instance or None
            A Leaf instance is returned if either `y` only contains one class
            label or if the maximum depth of the tree is reached. Otherwise,
            None is returned.
        """
        (y_unique, counts) = np.unique(y, return_counts=True)

        if len(y_unique) == 1:
            return Leaf(y_unique)

        if self.max_depth is None:
            return None
        elif depth >= self.max_depth:
            idx = np.argmax(counts)
            val = y_unique[idx]
            return Leaf(val)
        else:
            return None

    def eval_split(self, y_left, y_right):
        """Evaluate a candidate split.

        y_left : ndarray
            The true target value for one of the two partitions. Should be of
            shape (n_left,).
        y_right : ndarray
            The true target value for one of the two partitions. Should be of
            shape (n_right,).

        Returns
        -------
        float
            The value of the impurity measure for the given split.
        """
        p_left = np.mean(y_left, keepdims=True)
        p_right = np.mean(y_right, keepdims=True)
        if self.impurity_measure == 'gini':
            c_left = gini(p_left)
            c_right = gini(p_right)
        else:
            c_left = enropy(p_left)
            c_right = enropy(p_right)
        n_left = y_left.shape[0]
        n_right = y_right.shape[0]
        n = n_left + n_right
        return (n_left / n) * c_left + (n_right / n) * c_right

    def plot_boundaries(self, X, y):
        plt.figure()
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, style=y, legend=None)

        def plot_node(node):
            if isinstance(node, TreeNode):
                if node.feature == 0:
                    plt.axvline(x=node.split_val, color='r')
                else:
                    plt.axhline(y=node.split_val, color='r')
                plot_node(node.right_child)
                plot_node(node.left_child)

        plot_node(self.root)
        plt.show()


def gini(p):
    return 2 * p * (1 - p)


def enropy(p):
    return - (p * np.log(p) + (1 - p) * np.log(1 - p))


class TreeNode(object):
    """
    This implements a node in a tree which is not a leaf. It is used to store
    information about which feature to split and at which value.

    Attributes
    ----------
    feature : int
    split_val : float
    left_child : TreeNode instance or Leaf instance
    right_child : TreeNode instance or Leaf instance
    """
    def __init__(self, feature, split_val, left_child, right_child):
        self.feature = feature
        self.split_val = split_val
        self.left_child = left_child
        self.right_child = right_child


class Leaf(object):
    """Leaf node.

    This implements a leaf node in a tree which is used to store predictions.

    Attributes
    ----------
    pred : int
        The predicted label for an observation which ends up in this leaf.
    """
    def __init__(self, pred):
        self.pred = int(pred)
