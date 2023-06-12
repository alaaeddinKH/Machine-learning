import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Qt5Agg')
from graphviz import Digraph  # It was used to depict the tree
from sklearn.preprocessing import LabelEncoder
# -----------------------------data read and option-----------------------
df_train = pd.read_csv("trainSet.csv")
df_test = pd.read_csv("testSet.csv")
pd.set_option("display.max_columns", None)
# -----------------------------general check-------------------------------
df_train.head()
df_train.shape
df_train.dtypes
df_train.nunique()
df_test.head()
df_test.shape
df_train["class"].unique()
#  -----------------------split scale---------------------------------------

def gini_split(dataframe, feature, f_value, target):
    target_value = dataframe[target].unique()
    gini_n1 = 1
    gini_n2 = 1
    for i in target_value:
        c = round((dataframe[(dataframe[feature] == f_value) & (dataframe[target] == i)][feature].count()) / (dataframe[dataframe[feature] == f_value][feature].count()), 6)
        c2 = round((dataframe[(dataframe[feature] != f_value) & (dataframe[target] == i)][feature].count()) / (dataframe[dataframe[feature] != f_value][feature].count()), 6)

        gini_n1 -= c ** 2
        gini_n2 -= c2 ** 2
    n1_i_value = dataframe[dataframe[feature] == f_value][feature].count()
    n2_i_value = dataframe[dataframe[feature] != f_value][feature].count()
    n = len(dataframe)
    gin_split = round((((n1_i_value/n) * gini_n1) + ((n2_i_value/n) * gini_n2)), 6)
    return gin_split

#  -----------------------convert from  continuous feature to category---------------------------------------


def continuous_feature_check(dataframe):
    num_list = [col for col in dataframe.columns if (dataframe[col].nunique()>15) and (dataframe[col].dtype in ["float64", "int64"])]
    return num_list


def convert_to_category(dataframe, num_bins=5, drop_first=False):
    continuous_list = continuous_feature_check(dataframe)
    for col in continuous_list:
        feature_name = "new_" + str(col)
        dataframe[feature_name] = pd.cut(dataframe[col], bins=num_bins)

    if(drop_first):
        dataframe = dataframe.drop(continuous_list, axis=1)
    return dataframe


#  -----------------------class of node for used in tree great---------------------------------------


class Node:
    def __init__(self, feature, feature_value, row_count, class_value):
        self.feature = feature
        self.feature_value = feature_value
        self.left = None
        self.right = None
        self.row_count = row_count
        self.class_value = class_value

#  -----------------------class of Classification and regression TREE---------------------------------------


class CART:
    def __init__(self):  # attribute
        self.root = Node(None, None, 0, None)
        self.node_count = 0
        # self.used_feature = set()

    def fit(self, dataframe, target, depth, row_limit):
        self.root = self.great_tree(dataframe, target, depth, row_limit)

    def great_tree(self, dataframe, target, depth, row_limit):

        if len(dataframe) == 0:  # when all data finished split
            return Node(target, dataframe[target].value_counts().idxmax(), len(dataframe), dataframe[target].value_counts().idxmax())

        elif self.node_count > 2 ** depth:  # condition for look at tree depth and We take the most class when we have to divide before we get a net class
            return Node(target, dataframe[target].value_counts().idxmax(), len(dataframe),
                        dataframe[target].value_counts().idxmax())

        elif len(dataframe.columns) == 0:
            return Node(target, dataframe[target].value_counts().idxmax(), len(dataframe), dataframe[target].value_counts().idxmax())

        elif dataframe[target].nunique() == 1:  # When the data becomes only class
            return Node(target, dataframe[target].unique()[0], len(dataframe), dataframe[target].unique()[0])

        else:
            best_feature = None
            best_value = None
            best_gini = 0

            for f in dataframe.columns:
                # - if f in self.used_feature:
                    # continue
                for v in dataframe[f].unique():
                    if (dataframe[dataframe[f] == v][f].count() == 0) | (dataframe[dataframe[f] != v][f].count() == 0):
                        gini = 0
                        continue

                    gini = gini_split(dataframe, f, v, target)
                    if gini > best_gini:
                        best_feature = f
                        best_value = v
                        best_gini = gini
                # -self.used_feature.add(best_feature)

            if best_feature is None:
                return Node(target, dataframe[target].value_counts().idxmax(), len(dataframe), dataframe[target].value_counts().idxmax())

            node = Node(best_feature, best_value, len(dataframe), dataframe[target].value_counts().idxmax())
            self.node_count += 1
            if node.row_count <= row_limit:
                return Node(target, dataframe[target].value_counts().idxmax(), len(dataframe),dataframe[target].value_counts().idxmax())

            node.left = self.great_tree(dataframe[dataframe[best_feature] == best_value], target, depth, row_limit)
            node.right = self.great_tree(dataframe[dataframe[best_feature] != best_value], target, depth, row_limit)
            return node

    def print_tree(self, node: Node, depth=0):  # for print tree
        if node is None:
            return
        print("  |" * depth, end="")
        print("if " + str(node.feature) + " = " + str(node.feature_value) + " then class = " + str(
            node.class_value) + " " + str(node.row_count) + " ")
        self.print_tree(node.left, depth + 1)
        self.print_tree(node.right, depth + 1)

    def predict(self, target, data): # for prediction new data and output target of new data
        predictions = []
        for index, row in data.iterrows():
            node = self.root
            while node.feature != target:
                if row[node.feature] == node.feature_value:
                    node = node.left
                else:
                    node = node.right
            predictions.append(node.class_value)
        return predictions


def build_graph(node, do=None):  # with used Digraph library make graph for tree
    if do is None:
        do = Digraph()

    do.node(str(node), label=node.feature + " = " + str(node.feature_value) + " Row count " + str(node.row_count))

    if node.left is not None:
        do.edge(str(node), str(node.left), label="Yes")
        build_graph(node.left, do)

    if node.right is not None:
        do.edge(str(node), str(node.right), label="No")
        build_graph(node.right, do)

        return do


def evaluation(y, y_pre):  # for calculate evaluation scales
    tp = sum((y == 1) & (y_pre == 1))
    tn = sum((y == 0) & (y_pre == 0))
    fp = sum((y == 0) & (y_pre == 1))
    fn = sum((y == 1) & (y_pre == 0))
    tp_r = (tp/(tp+fp))
    tn_r = (tn/(tn+fn))
    acc = ((tp+tn)/(tp+fp+tn+fn))
    print("acc is =", acc)
    print("true positive count is =", tp)
    print("true positive rate is =", tp_r)
    print("true negative count is =", tn)
    print("true negative rate is =", tn_r)
    values = [acc, tp, tp_r, tn, tn_r]
    return values


def program_driver(df_train, df_test, target, depth, row_limit):  # Drive method
    """

    Parameters
    ----------
    df_train: train data for great model
    df_test: test data for prediction target of test data
    target: is target
    depth : Sets the depth of the tree to be built
    row_limit: The limit allocated for the number of lines, and if the number of lines is less, the branch is trimmed

    print: It prints the evaluation results for both the training and test data
    It exports a photograph of the tree in a PDF file
    It also exports test results as a text file
    -------

    """
    model = CART()
    df_train = convert_to_category(df_train, 5, True)  # convert numeric colum to category
    df_test = convert_to_category(df_test, 5, True)   # convert numeric colum to category
    model.fit(df_train, target, depth, row_limit)
    model.print_tree(model.root)
    dot = build_graph(model.root)
    dot.render("tree4", format="pdf")
    encoder = LabelEncoder()
    y_test = encoder.fit_transform(df_test[target])  # binary encoding
    y_test_pre = encoder.fit_transform(model.predict(target, df_test)) # binary encoding
    y_train = encoder.fit_transform(df_train[target]) # binary encoding
    y_train_pre = encoder.fit_transform(model.predict(target, df_train)) # binary encoding
    print("Eğitim (train) sonucu:")
    train_values = evaluation(y_train, y_train_pre)
    print("sınama (test) sonucu:")
    test_values = evaluation(y_test, y_test_pre)
    str_v = ["acc is", "true positive count is", "true positive rate is", "true negative count is", "true negative rate is"]
    with open('evaluation.txt', 'w', encoding='utf-8') as f:  # for export txt of evaluation values
        f.write("Eğitim (train) sonucu:\n")
        for i in range(len(train_values)):
            f.write(str(str_v[i]) + " :" + str(train_values[i]) + "\n")
        f.write("sınama (test) sonucu:\n")
        for i in range(len(test_values)):
            f.write(str(str_v[i]) + " :" + str(test_values[i]) + "\n")


program_driver(df_train, df_test, "class", 4, 4)
# We can change the depth and line stroke to get new instances of the tree







