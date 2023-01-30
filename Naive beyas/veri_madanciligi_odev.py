import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

# ########################### option ###################################
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# ###########################Data reading################################
df_train = pd.read_csv("trainSet.csv")
df_test = pd.read_csv("testSet.csv")

# ########################exploratory data analysis#######################


def control(df):
    print("general information for set")
    print("##############shape#################")
    print(df.shape)
    print("##############type#################")
    print(df.dtypes)
    print("##############discribe#################")
    print(df.describe().T)
    print("#################null value count##############")
    print(df.isnull().sum())
    print("##############unique values count#################")
    print(df.nunique())
    print("###############Head################")
    print(df.head())
    print("################Taill###############")
    print(df.tail())


# general information for train set
print("general information for train set")
control(df_train)
# general information for test set
print("general information for test set")
control(df_test)
# ######################analysis of variable's type##########################
num_list = [col for col in df_train.columns if df_train[col].nunique() > 10]
cat_list = [col for col in df_train.columns if df_train[col].nunique() <= 10]

# #########################categorical variable summery######################
for col in cat_list:
    print(pd.DataFrame({col: df_train[col].value_counts(),
                        "rate": 100 * df_train[col].value_counts() / len(df_train)}))

# #########################categorical variable graf######################
sns.countplot(x=df_train[cat_list[0]], data=df_train)
plt.show(block=True)

sns.countplot(x=df_train[cat_list[1]], data=df_train)
plt.show(block=True)

sns.countplot(x=df_train[cat_list[2]], data=df_train)
plt.show(block=True)

sns.countplot(x=df_train[cat_list[3]], data=df_train)
plt.show(block=True)

# ##################categorical variable summery with class(output variable)###########
for col in cat_list:
    print(pd.DataFrame({"target_count": df_train.groupby(col)["class"].value_counts()}), end="\n\n\n")

# ##############################check missing values##################################
# for train set
for col in df_train.columns:
    print(col, df_train[df_train[col] == '?'].any().sum())

# for train set
for col in df_test.columns:
    print(col, df_test[df_test[col] == '?'].any().sum())

# ############################## missing values solution#####################################
# solution num variable for train set (Replace the missing values with the arithmetic mean)
for col in num_list:
    df_train.loc[(df_train[col] == '?'), col] = 0
    # from string to float
    df_train[col] = df_train[col].astype(float)
    # from 0 to mean
    df_train.loc[(df_train[col] == 0), col] = df_train[col].mean()

# solution cat variable for train set (Delete entries that contain values?)
df_train = df_train[~(df_train["credit_history"] == '?')]
df_train.loc[(df_train["property_magnitude"] == '?'), "property_magnitude"] = 'no known property'

# solution num variable for test set
df_test = df_test[~(df_test["age"] == '?')]
df_test = df_test[~(df_test["credit_amount"] == '?')]
# solution cat variable for train set (Delete entries that contain values?)
df_test.loc[(df_test["property_magnitude"] == '?'), "property_magnitude"] = 'no known property'
df_test = df_test[~(df_test["credit_history"] == '?')]
df_test = df_test[~(df_test["employment"] == '?')]

# #######################numerical variable summery for train set########################

df_train["age"].describe().T
df_train["credit_amount"].describe().T

# #######################numerical variable summery for test set########################
df_test["age"].describe().T
df_test["credit_amount"].describe().T
# #######################numerical variable summery for train set - graf########################
# df_train["credit_amount"].hist()
        # plt.xlabel("credit_amount")
        # plt.xlim(1)
        # plt.ylim(2)
        # plt.title("credit_amount")
        # plt.show(block = True)

# df_train["age"].hist()
        # plt.xlabel("age")
        # plt.xlim(1)
        # plt.ylim(2)
        # plt.title("age")
        # plt.show(block = True)

# ####################numerical variable summery with class(output variable)############
for col in num_list:
    print(df_train.groupby("class").agg({col: ["mean", "max", "min"]}), end="\n\n\n")

# ############################See outlier values with graphs#################################
# for data train
# df_train["age"].plot(kind="box", vert=False)
# df_train["credit_amount"].plot(kind="box", vert=False)
# for data test
# df_test["age"].plot(kind="box", vert=False)
# df_test["credit_amount"].plot(kind="box", vert=False)

# ##########################One hat encoding for train set#######################################
cat_list1 = [col for col in df_train.columns if (df_train[col].nunique() <= 10) and (df_train[col].nunique() > 2)]
# for train data
df_train_prep3 = df_train
for col in cat_list1:
    dummies = pd.get_dummies(df_train_prep3[col])
    df_train_prep3 = pd.concat([df_train_prep3, dummies], axis="columns")
    df_train_prep3.drop([col], axis="columns", inplace=True)
df_train_prep3.head()
# for test data
df_test_prep4 = df_test
df_test_prep4.head()
for col in cat_list1:
    dummies = pd.get_dummies(df_test_prep4[col])
    df_test_prep4 = pd.concat([df_test_prep4, dummies], axis="columns")
    df_test_prep4.drop([col], axis="columns", inplace=True)
df_test_prep4.head()

# ####################################binary encoding for output variable for train set#####################
coder = LabelEncoder()
df_train_prep3["class"] = coder.fit_transform(df_train_prep3["class"])
df_train_prep3["class"].value_counts()
df_train_prep3["class"].head()
# ####################################binary encoding for output variable for test set#####################
coder = LabelEncoder()
df_test_prep4["class"] = coder.fit_transform(df_test_prep4["class"])
df_test_prep4["class"].value_counts()

# #######################################model##################################
y_train = df_train_prep3["class"]
x_train = df_train_prep3.drop("class", axis="columns")
y_test = df_test_prep4["class"]
x_test = df_test_prep4.drop("class", axis="columns")

model2 = GaussianNB()
model2.fit(x_train, y_train)
model2.score(x_test, y_test)


# ########################################Accuracy function#################################
def evaluation(y, y_pre):
    tp = sum((y == 1) & (y_pre == 1))
    tn = sum((y == 0) & (y_pre == 0))
    fp = sum((y == 0) & (y_pre == 1))
    fn = sum((y == 1) & (y_pre == 0))
    tp_r = (tp/(tp+fp))
    tn_r = (tn/(tn+fn))
    acc = ((tp+tn)/(tp+fp+tn+fn))
    print("test results is")
    print("acc is =", acc)
    print("true positive count is =", tp)
    print("true positive rate is =", tp_r)
    print("true negative count is =", tn)
    print("true negative rate is =", tn_r)


y_predicted = model2.predict(x_test)

evaluation(y_test, y_predicted)
