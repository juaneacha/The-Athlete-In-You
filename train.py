import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.colors import ListedColormap
import seaborn as sns
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
df1 = pd.read_csv("D://Download//athlete_events.csv")#, index_col = True)
df1 = df1.loc[:,['Sex','Age', 'Height', 'Weight', 'Sport', 'Team', 'Year', 'Season']]


#CLEANING DATA

# Identify the outliers
outliers = np.where(np.abs(df1 - df1.mean()) > 3 * df1.std())[0]

# Remove the outliers
df1 = df1.drop(outliers)
df1 = df1.reset_index(drop=True)

#Imputation and removable of NA
df1['Team'] = df1['Team'].fillna('No Team')
df = df1
df = df.dropna()

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



#MACHINE LEARNING MODELING

# Split data into dependent/independent variables
y= df['Sport']#_net.iloc[:, :-1].values
x= df.drop(['Sport'], axis = 1)#_net.iloc[:, -1].values


# Split data into test/train set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = True)


"""# Scale dataset
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)"""


#Random Forest
from sklearn.ensemble import RandomForestClassifier

#Create Random Forest Classifier
rf = RandomForestClassifier()

#Train The Model
rf.fit(X_train, y_train)

#Test The Model 
y_predRF = rf.predict(X_test)

#Saving the Model
import pickle
pickle.dump(rf, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))

