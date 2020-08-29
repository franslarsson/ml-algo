
import numpy as np

from decision_tree import DecisionTree, TreeNode, ClassificationTree, Leaf


def test_split_one_feature():
    """Test implementation of `find_split`."""
    tree = ClassificationTree()

    X1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y1 = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    _, split_val = tree.find_split(X1, y1)
    assert split_val == (4 + 5) / 2

    X2 = np.array([3, 10, 4, 1, 9, 7, 6, 8, 5, 2])
    y2 = np.array([0, 1, 0, 0, 1, 1, 1, 1, 1, 0])
    _, split_val = tree.find_split(X2, y2)
    assert split_val == (4 + 5) / 2


def test_first_find_feature():
    """Test the implementation of `find_feature`."""
    tree = ClassificationTree()

    X = np.array([[ .1,   .5],
                  [ .3 ,  .7],
                  [0.2,  1.9],
                  [0.1,  2.3],
                  [0.4,  0.3]])
    y = np.array([1, 1, 0, 0, 1])
    f, split_val = tree.find_feature(X, y)
    assert (f, split_val) == (1, (0.7 + 1.9)/2)


def test_partition():
    """Test the implementation of `find_feature`."""
    tree = ClassificationTree()

    X = np.array([[1,  1],
                   [2,  1],
                   [1,  2],
                   [2,  2],
                   [2,  3]])
    y = np.array([1, 1, 0, 0, 1])
    root = tree.partition(X, y)
    assert root.split_val == 1.5
    assert root.feature == 1
    assert root.left_child.pred == 1
    assert root.right_child.split_val == 2.5
    assert root.right_child.feature == 1
    assert root.right_child.right_child.pred == 1
    assert root.right_child.left_child.pred == 0


def test_predict():
    """Test the implementation of `find_feature`."""
    tree = ClassificationTree()

    X1 = np.array([[1, 1],
                   [2,  1],
                   [1,  2],
                   [2,  2],
                   [2,  3]])
    y1 = np.array([1, 1, 0, 0, 1])
    tree.fit(X1, y1)
    assert tree.predict(np.array([3, 0.5])) == 1
    assert tree.predict(np.array([10, 10])) == 1
    assert tree.predict(np.array([5, 1.51])) == 0


def test_max_depth():
    """Test the implementation of `max_depth` and `check_partition`."""
    tree = ClassificationTree(max_depth=1)

    X = np.array([[1, 1],
                  [2,  1],
                  [1,  2],
                  [2,  2],
                  [2,  3]])
    y = np.array([1, 1, 0, 0, 1])
    tree.fit(X, y)
    assert tree.root.split_val == 1.5
    assert tree.root.feature == 1
    assert isinstance(tree.root.left_child, Leaf)
    assert isinstance(tree.root.right_child, Leaf)
    assert tree.root.left_child.pred == 1
    assert tree.root.right_child.pred == 0
    assert tree.predict(np.array([2, 3])) == 0
