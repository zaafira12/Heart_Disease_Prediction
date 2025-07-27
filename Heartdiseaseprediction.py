import pandas as pd
from sklearn.impute import SimpleImputer 
data = pd.read_csv('heart_statlog_cleveland_hungary_final.csv')
data.head()
data.tail()
print('Missing Values:')
print(data.isnull().sum())

import seaborn as sns
import matplotlib.pyplot as plt
# Plotting target variable distribution
sns.countplot(x='target', hue='target', data=data, palette='Set2', legend=False)
plt.title('Heart Disease Distribution')
plt.xlabel('Heart Disease (1 = Yes, 0 = No)')
plt.ylabel('Count')
plt.grid(axis='y', linestyle='--', alpha=0.7)
print(data['target'].value_counts())
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Drop target and calculate feature means
feature_df = data.drop(columns='target')
feature_means = feature_df.mean()
features = feature_means.index
means = feature_means.values
# Y positions
y_pos = np.arange(len(features))
# Colors
colors = plt.cm.rainbow(np.linspace(0, 1, len(features)))
# Determine suitable x-axis range
x_min = 0
x_max = max(means) + 5  # add a small margin for labels
# Plot
plt.figure(figsize=(10, 7))
for i in range(len(features)):
    plt.hlines(y=y_pos[i], xmin=0, xmax=means[i], color=colors[i], linewidth=6)
# Annotations
for i in range(len(features)):
    plt.text(means[i] + 0.3, y_pos[i], f'{means[i]:.2f}', va='center', fontsize=9)
# Aesthetics
plt.yticks(y_pos, features)
plt.xlim(x_min, x_max)
plt.xlabel('Mean Value')
plt.title('Feature-wise Mean Value (Horizontal Colorful Line Plot)')
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


labels = ['Training Set (80%)', 'Testing Set (20%)']
sizes = [80, 20]  # You can also use actual counts if preferred
colors = ['#FFC4C4', '#B2C8BA']  # Soft nude colors
explode = (0.05, 0.05)  # Slight separation between slices

# Create pie chart
plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, colors=colors, explode=explode, autopct='%1.1f%%',
        startangle=90, shadow=True, textprops={'fontsize': 12})
plt.title('Train-Test Split of Dataset', fontsize=14)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.show()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
from sklearn.preprocessing import StandardScaler

X = data.drop(columns = ['target'])
Y = data['target']
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators = 100,random_state =42),
    "Support Vector Machine": SVC(probability = True),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors = 5)
}

def evaluate_model(model,X_train,X_test,Y_train,Y_test):
    model.fit(X_train,Y_train)
    Y_pred = model.predict(X_test)
    Y_proba = model.predict_proba(X_test)[:,1] if hasattr(model,"predict_proba")else None

    accuracy = accuracy_score(Y_test,Y_pred)
    precision = precision_score(Y_test,Y_pred)
    recall = recall_score(Y_test,Y_pred)
    f1 = f1_score(Y_test,Y_pred)
    roc_auc = roc_auc_score(Y_test,Y_proba)if Y_proba is not None else "N/A"

    return accuracy,precision,recall,f1,roc_auc

results = {}
for name,model in models.items():
    accuracy,precision,recall,f1,roc_auc = evaluate_model(model,X_train,X_test,Y_train,Y_test)
    results[name]={
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC_AUC": roc_auc
    }

results_df = pd.DataFrame(results).T
print(results_df)

# Example model names and their accuracies
models = [  'Logistic Regression','Random Forest ','SVM','KNN']
accuracies = [0.84, 0.92, 0.87, 0.83]  # Replace these with your actual values
# Choose attractive, balanced colors
colors = ['#F2B5D4', '#A2D5F2', '#D5E1DF', '#F9D5E5']
# Plotting a horizontal bar chart
plt.figure(figsize=(8, 5))
plt.barh(models, accuracies, color=colors)
plt.xlabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.xlim(0, 1.05)  # Accuracy is between 0 and 1
for i, v in enumerate(accuracies):
    plt.text(v + 0.01, i, f"{v:.2f}", va='center', fontsize=10)

plt.tight_layout()
plt.show()

from sklearn.model_selection import StratifiedKFold,cross_val_score
df = pd.read_csv('heart_statlog_cleveland_hungary_final.csv')
X = df.drop(columns=['target'])
Y = df['target']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
rf = RandomForestClassifier(n_estimators = 100,random_state = 42)
skf = StratifiedKFold(n_splits = 5,shuffle = True,random_state = 42)
cv_scores = cross_val_score(rf,X_scaled,Y,cv = skf,scoring = 'accuracy')
print("Cross-Validation Accuracy Scores:",cv_scores)
print("Mean Accuracy:",np.mean(cv_scores)*100)
print("Standard Deviation:",np.std(cv_scores))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100,random_state=42)
train_sizes,train_scores,cv_scores = learning_curve(
    rf_model,X_train,Y_train,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    train_sizes = np.linspace(0.92,0.1,10),
)
train_scores_mean = np.mean(train_scores,axis=1)
train_scores_std = np.std(train_scores,axis=1)
cv_scores_mean = np.mean(cv_scores,axis=1)
cv_scores_std = np.std(cv_scores,axis =1)

plt.figure(figsize=(8,4))
plt.plot(train_sizes,train_scores_mean,'o-',color ="r",label="Traning score")
plt.fill_between(train_sizes,train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std,alpha=0.92,color="r")
plt.plot(train_sizes,cv_scores_mean,'o-',color="g",label="Cross-Validation score")
plt.fill_between(train_sizes,cv_scores_mean - cv_scores_std,
                 cv_scores_mean + cv_scores_std,alpha =0.1,color="g")
plt.title("Learning curve for Random Forest")
plt.xlabel("Traning Examples")
plt.ylabel("Accuracy Score")
plt.legend(loc="best")
plt.grid(True)
plt.show()


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# Define models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC(),
    "K-Nearest Neighbors": KNeighborsClassifier()
}

# Plot confusion matrices
plt.figure(figsize=(14, 10))
for i, (name, model) in enumerate(models.items(), 1):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.subplot(2, 2, i)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

plt.tight_layout()
plt.show()


    # building a predictive system 
input_data=(48,0,2,120,284,0,0,120,0,0.0,1)

# change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predciting for only 1 
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if(prediction[0]==0):
    print("The person does not have Heart Disease")
else:
    print("The person has Heart Disease")    