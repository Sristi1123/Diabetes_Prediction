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
