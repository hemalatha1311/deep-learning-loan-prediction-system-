import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
# Build path relative to this script so it works regardless of cwd
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.normpath(os.path.join(script_dir, os.pardir, "dataset", "loan_data.csv"))
data = pd.read_csv(data_path)
print(data.head())

# Convert categorical values
# Drop identifiers that aren't predictive
if 'Loan_ID' in data.columns:
    data = data.drop('Loan_ID', axis=1)

# Convert yes/no and other binary categories
data['Gender'] = data['Gender'].map({'Male':1,'Female':0})
data['Married'] = data['Married'].map({'Yes':1,'No':0})
data['Education'] = data['Education'].map({'Graduate':1,'Not Graduate':0})
data['Self_Employed'] = data['Self_Employed'].map({'Yes':1,'No':0})
data['Loan_Status'] = data['Loan_Status'].map({'Y':1,'N':0})

# Dependents column has values like '3+' and strings; convert to numeric
if 'Dependents' in data.columns:
    data['Dependents'] = data['Dependents'].replace('3+', '3')
    data['Dependents'] = pd.to_numeric(data['Dependents'], errors='coerce')

# Property area mapping
if 'Property_Area' in data.columns:
    data['Property_Area'] = data['Property_Area'].map({'Rural':0,'Semiurban':1,'Urban':2})

# Fill or drop remaining missing values
data = data.fillna(0)

X = data.drop('Loan_Status', axis=1)
y = data['Loan_Status']

# split data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

# scale features to help convergence
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# train model with more iterations and specify solver
model = LogisticRegression(max_iter=500, solver='lbfgs')
model.fit(X_train,y_train)

# save both scaler and model for later use
with open("loan_model.pkl","wb") as f:
    pickle.dump({"model": model, "scaler": scaler}, f)

# report performance
accuracy = model.score(X_test, y_test)
print(f"Model trained successfully, test accuracy: {accuracy:.4f}")