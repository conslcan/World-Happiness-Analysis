# import kagglehub
# # Download latest version
# path = kagglehub.dataset_download("unsdsn/world-happiness")
# print("Path to dataset files:", path)

import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns  
import sklearn 
from sklearn.metrics import mean_absolute_error, r2_score
print(sklearn.__version__)
# Load the dataset
df = pd.read_csv("data/world_happiness.csv")  

# Print data types to check for non-numeric columns
# print(df.dtypes)

# Check for missing values
# print(df.isnull().sum())  

# Rename the column 'Happiness.Score' to 'Happiness_Score'
df.rename(columns={'Happiness.Score': 'Happiness_Score'}, inplace=True)

# Select only numeric columns for correlation calculation
df_numeric = df.select_dtypes(include=['float64', 'int64'])

# Correlation heatmap
# plt.figure(figsize=(10, 6))
# sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm')
# plt.title("Correlation Between Happiness and Other Factors")
# plt.show()

from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_squared_error  
X = df[['GDP per capita', 'Social support', 'Healthy life expectancy']]  
y = df['Score']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
model = LinearRegression()  
model.fit(X_train, y_train)  
y_pred = model.predict(X_test)  
print("Model Accuracy:", model.score(X_test, y_test))  
print("RMSE:", mean_squared_error(y_test, y_pred))

