import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

#LOADING DATA
df = pd.read_csv("D://Download//athlete_events.csv") #Source: https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-history-athletes-and-results
df = df.loc[:,['Sex','Age', 'Height', 'Weight', 'Sport', 'Team', 'Season']]

#Imputation and removable of NA
df['Team'] = df['Team'].fillna('No Team')
df = df.dropna()

#Casting Strings to Int
def convertRespToInt(df, feature):
    tempFeature = df[feature].unique()

    i = 0
    while i < len(tempFeature):
        df[feature] = df[feature].replace(tempFeature[i], i)
        i += 1

    df[feature] = df[feature].astype(float)
    df[feature].unique()

convertRespToInt(df, 'Sex')
convertRespToInt(df, 'Sport')
convertRespToInt(df, 'Season')
convertRespToInt(df, 'Team')

#Dropping Outliers
def drop_outliers_IQR(df):

   q1=df.quantile(0.25)

   q3=df.quantile(0.75)

   IQR=q3-q1

   not_outliers = df[~((df<(q1-1.5*IQR)) | (df>(q3+1.5*IQR)))]

   outliers_dropped = not_outliers.dropna()

   return outliers_dropped

df = drop_outliers_IQR(df)

#Balancing of classes
bal_df = pd.DataFrame()
sport_series = df['Sport'].unique()

for s in sport_series:
    try:
        temp_df = df[df['Sport'] == s].sample(4000)
        bal_df = pd.concat([bal_df, temp_df])
    except:
        continue

df = bal_df



#MACHINE LEARNING MODELING

# Split data into dependent/independent variables
y= df['Sport']
x= df.drop(['Sport'], axis = 1)

# Split data into test/train set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = True)

"""# Scale dataset
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
"""

"""#Random Forest
from sklearn.ensemble import RandomForestClassifier

#Create Random Forest Classifier
rf = RandomForestClassifier()

#Train The Model
rf.fit(X_train, y_train)
"""

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)



#Saving the Model
import pickle
pickle.dump(clf, open('model_dt.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))
