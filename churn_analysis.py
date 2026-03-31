import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,f1_score,recall_score
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
df = pd.read_excel("Telco_customer_churn1.xlsx")
#FIRST 5 ROW
print("First 5 rows")
print(df.head())

#COLUNMS NAMES
print("\ncolumns name")
print(df.columns)

#Data info
print("\ninfo")
print(df.info)

#Check Null Values
print("\nNull values")
print(df.isnull().sum())

##===DATA CLEANING===
# ===================


#Checking Totalcharges datatype
print("\nDatatype Before fix")
print(df.dtypes)

#Convert TotalCharge to Numeric
df["Total Charges"] = pd.to_numeric(df["Total Charges"],errors="coerce")

#Checking Again
print("\nDatatype After fix")
print(df.dtypes)

#Check Null Values
print("\nNull values")
print(df.isnull().sum())


#Removing Useless Columns
df = df.drop(["CustomerID","Count","Country","State","City","Zip Code","Lat Long","Latitude","Longitude","Churn Reason","Churn Score","Churn Label"],axis=1)
print("df arter dropping columns:",df)

# Colums After Droping
print("\n columns after droping")
print(df.columns)


# Droping Null Values
df = df.dropna()


#Checking Null Values After Droping

print("\nAfter Droping Null Values")
print(df.isnull().sum())


#Checking Targeted Column
print("\nChurn Values:")
print(df['Churn Value'].value_counts())

#Checking Datatype
print(df.dtypes)


#Encoding
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
df["Partner"] = df["Partner"].map({"Yes": 1, "No": 0})
df['Dependents'] = df['Dependents'].map({'Yes': 1, 'No': 0})
df["Phone Service"] = df["Phone Service"].map({"Yes": 1, "No": 0})
df["Paperless Billing"] = df["Paperless Billing"].map({"Yes": 1, "No": 0})
df["Senior Citizen"] = df["Senior Citizen"].map({"Yes": 1,"No": 0})
print(df.head)

#Converting Str Into Numeric
df = pd.get_dummies(df,drop_first=True)

#Converting Bool Into Int
bool_cols = df.select_dtypes(include="bool").columns
df[bool_cols] = df[bool_cols].astype(int)


#Checking Datatype again
print(df.dtypes)

#Features & Target split
X = df.drop("Churn Value",axis=1)
y = df["Churn Value"]

#Train Test Split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


#standdize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Training the model
model = RandomForestClassifier(n_estimators=300,max_depth=10,min_samples_split=5,
                               min_samples_leaf=2,max_features="sqrt",class_weight="balanced",random_state=42)
model.fit(X_train,y_train)


#Prediction and accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print("result by randomforest")
print("\nModel Accuracy:",accuracy)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
#visualisation y_test vs y_predict
plt.scatter(x=y_test,y=y_pred)
plt.title("test vs train")
plt.show()

#logisticRegression
model = LogisticRegression(class_weight="balanced",max_iter=1000)
model.fit(X_train,y_train)
y_prob = model.predict_proba(X_test)[:,1]
y_pred = (y_prob > 0.3).astype(int)
print("\nThresold:",0.3)
print("\nRecall:",recall_score(y_test,y_pred))
f1 = f1_score(y_test,y_pred)
print("F1 score:",f1)
accuracy = accuracy_score(y_test,y_pred)
print("result by logistic regression")
print("\nModel Accuracy:",accuracy)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# pickling the best model
pickle.dump(model,open("model.pkl","wb"))
pickle.dump(X.columns,open("columns.pkl","wb"))


#Visualisation
sns.countplot(x="Churn Value",data=df)
plt.title("Churn Disturbution")
plt.show()

#Monthly charges VS Churn
sns.boxplot(x="Churn Value",y="Monthly Charges",data=df)
plt.title("Monthly Charges VS Churn Value")
plt.show()

#contract Type Vs Churn value
sns.countplot(x="Contract_One year",hue="Churn Value",data=df)
plt.title("contract VS Churn Value")
plt.show()



sns.countplot(x="Contract_Two year",hue="Churn Value",data=df)
plt.title("contract VS Churn Value")
plt.show()



