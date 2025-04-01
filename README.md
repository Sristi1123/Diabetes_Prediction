# Diabetes_Prediction
#🚀 Diabetes Prediction using Machine Learning (KNN Model) This project uses K-Nearest Neighbors (KNN) to predict diabetes based on health parameters. It includes data #preprocessing, visualization, model training, hyperparameter tuning, and performance evaluation.


#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from mlxtend.plotting import plot_decision_regions
import missingno as msno
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
# Load dataset
df=pd.read_csv("/content/diabetes.csv")
df.head()
df.columns
df.info() # Print dataset information
df.describe()
df.describe().T  # Summary statistics
df.isnull()
df.isnull().sum()   # Check for missing values
# Create a deep copy of the dataset for processing
df_copy=df.copy(deep=True)
# Replace zero values with NaN in specific columns to handle missing data
df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']]=df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.nan)
df_copy.isnull().sum()
# Plot histograms before data cleaning
p=df.hist(figsize=(20,20))
# Fill missing values with mean or median
df_copy['Glucose'].fillna(df_copy['Glucose'].mean(),inplace=True)
df_copy['BloodPressure'].fillna(df_copy['BloodPressure'].mean(),inplace=True)
df_copy['SkinThickness'].fillna(df_copy['SkinThickness'].median(),inplace=True)
df_copy['Insulin'].fillna(df_copy['Insulin'].median(),inplace=True)
df_copy['BMI'].fillna(df_copy['BMI'].median(),inplace=True)
# plot histograms after data cleaning
p=df_copy.hist(figsize=(20,20))
# Visualize missing values using MissingNo library
p=msno.bar(df_copy)
import matplotlib.pyplot as plt

# Define colors for the bars
color_wheel = {0: "#0392cf", 1: "#7bc043"}  # Blue for Non-Diabetic, Green for Diabetic

# Count occurrences of each Outcome (0 and 1)
outcome_counts = df.Outcome.value_counts()

# Rename the index (0 → Non-Diabetic, 1 → Diabetic)
outcome_counts.index = ["Non-Diabetic", "Diabetic"]

# Plot bar chart with updated labels
p = outcome_counts.plot(kind="bar", color=[color_wheel[0], color_wheel[1]])

# Set axis labels
p.set_xlabel("Outcome")
p.set_ylabel("Frequency")

# Keep labels straight
plt.xticks(rotation=0)

plt.show()
 # scatter matrix to show relationships between numerical features.
p=scatter_matrix(df,figsize=(20,20))
# Seaborn for better visualization
p=sns.pairplot(df_copy,hue='Outcome')
# Correlation heatmap befire data cleaning
plt.figure (figsize=(12,10))
p=sns.heatmap(df.corr(),annot=True,cmap="YlGnBu")
# Correlation heatmap after data cleaning
plt.figure(figsize=(12,10))
p=sns.heatmap(df_copy.corr(),annot=True,cmap="YlGnBu")
# print first 5 rows
df_copy.head()
# Standardize feature values for better KNN performance
sc_x=StandardScaler()
x=pd.DataFrame(sc_x.fit_transform(df_copy.drop(["Outcome"],axis=1)),columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age'])
x.head()
y=df_copy.Outcome
y
# Split data into training and testing sets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=42,stratify=y)
# Find the best K value using training and testing scores
test_score=[]
train_score=[]
for i in range(1,15):
  knn=KNeighborsClassifier(n_neighbors=i)
  knn.fit(x_train,y_train)
  train_score.append(knn.score(x_train,y_train))
  test_score.append(knn.score(x_test,y_test))
train_score
train_score
# Print best K values
max_train_score=max(train_score)
train_score_ind=[i for i,v in enumerate(train_score) if v==max_train_score]
print("Max train score {} % and k = {}".format(max_train_score*100,list(map(lambda x:x+1,train_score_ind))))
max_test_score=max(test_score)
test_score_ind=[i for i,v in enumerate(test_score) if v==max_test_score]
print("Max test score {} % and k = {}".format(max_test_score*100,list(map(lambda x:x+1,test_score_ind))))
# Plot accuracy vs number of neighbors (K)
plt.figure(figsize=(12,5))
plt.plot(range(1,15),test_score,color="blue",label="Testing Accuracy")
plt.plot(range(1,15),train_score,color="red",label="Training Accuracy")
plt.legend()
# Train KNN model with optimal K=11
knn=KNeighborsClassifier(n_neighbors=11)
knn.fit(x_train,y_train)
knn.score(x_test,y_test)
# Plot decision regions
value = 20000
width = 20000

plot_decision_regions(x.values, y.values, clf=knn, legend=2,
                      filler_feature_values={2: value, 3: value, 4: value, 5: value, 6: value, 7: value},
                      filler_feature_ranges={2: width, 3: width, 4: width, 5: width, 6: width, 7: width},
                      X_highlight=x_test.values)

plt.title('KNN with Diabetes Dataset')
plt.show()

import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

# Select only two features (e.g., Glucose and BMI)
x = x[['Glucose', 'BMI']]  # This selects only 'Glucose' and 'BMI' columns
y = y  # Keep the target variable (Outcome)

# Fit the KNN model (already done in your previous steps)
knn.fit(x, y)

# Plot the decision regions for these two features
plot_decision_regions(X=x.values, y=y.values, clf=knn, legend=2)

# Title for the plot
plt.title('KNN with Diabetes Dataset')
plt.show()
# Predictions on test set
y_pred=knn.predict(x_test)
# Generate confusion matrix and heatmap
cnf_matrix=metrics.confusion_matrix(y_test,y_pred)
p=sns.heatmap(pd.DataFrame(cnf_matrix),annot=True,cmap="YlGnBu")
plt.title("Confusion matrix",y=1.1)
plt.ylabel("Actual label")
plt.xlabel("Predicted label")

# Classification report
print(classification_report(y_test,y_pred))

import joblib
joblib.dump(knn, "knn_model.pkl")
joblib.dump(sc_x, "scaler.pkl")

# =============================================
# INTERACTIVE PREDICTION WIDGET
# =============================================
import ipywidgets as widgets
from IPython.display import display, clear_output

# Define sliders for user input
pregnancies = widgets.IntSlider(min=0, max=20, value=1, description="Pregnancies")
glucose = widgets.IntSlider(min=50, max=200, value=100, description="Glucose")
bp = widgets.IntSlider(min=40, max=150, value=70, description="Blood Pressure")
skin = widgets.IntSlider(min=0, max=100, value=20, description="Skin Thickness")
insulin = widgets.IntSlider(min=0, max=900, value=30, description="Insulin")
bmi = widgets.FloatSlider(min=10, max=60, value=25.0, description="BMI")
dpf = widgets.FloatSlider(min=0.0, max=3.0, value=0.5, description="DPF")
age = widgets.IntSlider(min=10, max=100, value=30, description="Age")

# Output area
output = widgets.Output()

# Function to predict diabetes
def predict_diabetes(b):
    with output:
        clear_output(wait=True)

        user_data = np.array([[pregnancies.value, glucose.value, bp.value, skin.value,
                             insulin.value, bmi.value, dpf.value, age.value]])
        user_data_scaled = sc_x.transform(user_data)
        prediction = knn.predict(user_data_scaled)[0]
        probability = knn.predict_proba(user_data_scaled)[0][1]

        # Set emoji-based feedback
        if prediction == 1:
            emoji = "⚠️"
            message = f"{emoji} High Diabetes Risk! ({probability:.2%})"
            color = "red"
        else:
            emoji = "🎉"
            message = f"{emoji} Low Diabetes Risk! ({probability:.2%})"
            color = "green"

        # Show result
        print(message)

        # Enhanced Visual Feedback (Gradient Progress Bar)
        fig, ax = plt.subplots(figsize=(6, 1.2))
        ax.barh([""], [probability], color=sns.color_palette("coolwarm", as_cmap=True)(probability), height=0.5)
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.text(probability/2, 0, f"{probability:.1%}", va="center", ha="center",
                fontsize=12, color="white", weight="bold")

        plt.title("Diabetes Risk Probability", fontsize=14, weight="bold", color=color)
        plt.show()

# Create a button
button = widgets.Button(description="🚀 Predict Now")
button.on_click(predict_diabetes)

# Display UI
print("\n\n=== Diabetes Risk Prediction Tool ===")
display(pregnancies, glucose, bp, skin, insulin, bmi, dpf, age, button, output)
