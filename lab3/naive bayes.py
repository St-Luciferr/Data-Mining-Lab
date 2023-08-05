import pandas as pd

# Load the dataset into a DataFrame
df=pd.read_csv('dataset/BankChurners.csv')

df.head(5)

new_column_names = {
    'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2': 'mon_2',
    'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1': 'mon_1',
}

df.rename(columns=new_column_names, inplace=True)
df.head()

df.drop(columns=['CLIENTNUM','mon_1','mon_2'],inplace=True)

print(df.info())
print(df.head(5))
print(df.isna().sum())
print(df.duplicated().sum())

# Display count of survivors (0: Not Survived, 1: Survived)
label_counts = df['Attrition_Flag'].value_counts()
print(label_counts)

type(label_counts)

import seaborn as sns
import matplotlib.pyplot as plt
# Create the count plot
sns.countplot(x='Attrition_Flag', data=df)

text_box = {
    'boxstyle': 'round',
    'facecolor': 'white',
    'alpha': 0.8
}

plt.text(1.1, 8000, 'THA076BCT026\nTHA076BCT027\nTHA076BCT041', fontsize=8, bbox=text_box, color='red')
# Set plot labels
plt.xlabel('Atrition Flag')
plt.ylabel('Count')
plt.title('Number of Customers')
plt.tight_layout()
plt.savefig('Number of Label.png',bbox_inches='tight')
# Show the plot
plt.show()

df.replace({'Attrition_Flag':{'Existing Customer':0,'Attrited Customer':1}},inplace=True)
# Group the data by 'Sex' and calculate the mean of 'Attrition_Flag' for each group
Atrition_rate_by_gender = df.groupby('Gender')['Attrition_Flag'].mean()

# Print the Atrition rates
print(Atrition_rate_by_gender)

# Create a pivot table to calculate the mean of 'Attrition_Flag' for each combination of 'Sex' and 'Card_Category'
Attrition_rate_pivot = df.pivot_table(values='Attrition_Flag', index='Gender', columns='Card_Category', aggfunc='mean')

# Print the survival rates
print(Attrition_rate_pivot)

# Plot the survival rates using bar function
Attrition_rate_pivot.plot(kind='barh', figsize=(10, 5))
plt.xlabel('Attrition Rate')
plt.ylabel('Gender')
plt.title('Attrition Rate by Gender and Card_Category')
plt.legend(title='Card_Category')
plt.text(0.36, 1.25, 'THA076BCT026\nTHA076BCT027\nTHA076BCT041', fontsize=8, bbox=text_box, color='red')
plt.tight_layout()
plt.savefig('Attrition Rate by Gender and Card_Category.png',bbox_inches='tight')
plt.show()

# Discretize the 'Age' column into age groups using the cut function
age_bins = [0, 18, 30, 50, 100]
age_labels = ['0-17', '18-29', '30-49', '50+']
df['Age_Group'] = pd.cut(df['Customer_Age'], bins=age_bins, labels=age_labels)

df.head(5)

# Create a pivot table to calculate the mean of 'Attrition_Flag' for each combination of 'Sex', 'Age_Group', and 'Card_Category'
Attrition_rate_pivot = df.pivot_table(values='Attrition_Flag', index=['Gender', 'Age_Group'], columns='Card_Category', aggfunc='mean')

# Plot the Attrition rates using barh function
Attrition_rate_pivot.plot(kind='barh', figsize=(10, 8))
plt.xlabel('Attrition Rate')
plt.ylabel('Gender and Age Group')
plt.title('Attrition Rate by Gender, Age Group, and Card_Category')
plt.legend(title='Card_Category', loc='upper right')
plt.text(0.38, 5, 'THA076BCT026\nTHA076BCT027\nTHA076BCT041', fontsize=8, bbox=text_box, color='red')
plt.tight_layout()
plt.savefig('Attrition Rate by Gender, Age Group, and Card_Category.png',bbox_inches='tight')
plt.show()

# Create a scatter plot for each passenger class
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Credit_Limit', y='Card_Category', hue='Card_Category', palette='Set1', s=100, alpha=0.7)

plt.title('Credit Limit for Each Category')
plt.legend()
plt.text(32000, 0.32, 'THA076BCT026\nTHA076BCT027\nTHA076BCT041', fontsize=8, bbox=text_box, color='red')
plt.tight_layout()
plt.savefig('Credit Limit for Each Category.png',bbox_inches='tight')

# Show the plot
plt.show()

# Create a new DataFrame without the 'Class' variable
df1 = df.drop('Card_Category', axis=1)
df1.head(5)
numeric_df = df1.replace({'M': 1, 'F': 0})
numeric_df.head()

numeric_df.describe()

new_df=numeric_df.drop(columns=['Marital_Status','Education_Level','Income_Category','Age_Group'])
# Assuming 'df' is the DataFrame containing the data
correlation_matrix = new_df.corr(method='pearson')

# Create a heatmap using Seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Pearson Correlation Heatmap')
plt.text(.5, -0.1, 'THA076BCT026\nTHA076BCT027\nTHA076BCT041', fontsize=8, bbox=text_box, color='red')
plt.tight_layout()
plt.savefig('Pearson Correlation Heatmap.png',bbox_inches='tight')
plt.show()
correlation_with_target=correlation_matrix.iloc[0, 1:]

# Plot Correlation with Target
plt.figure(figsize=(8, 5))
sns.barplot(x=correlation_with_target.index, y=correlation_with_target.values,)
plt.title('Correlation to Attrition Flag')
plt.xlabel('Attribute')
plt.ylabel('Correlation')
plt.text(12, 0.17, 'THA076BCT026\nTHA076BCT027\nTHA076BCT041', fontsize=8, bbox=text_box, color='red')
plt.tight_layout()
plt.xticks(rotation=90)
plt.savefig('Correlation to Attrition Flag.png',bbox_inches='tight')
plt.show()

# Get the list of column names
column_names = new_df.columns.tolist()
print(column_names)

new_df.hist(bins=50, figsize=(20, 13),color='steelblue')
plt.text(0.8, 2100, 'THA076BCT026\nTHA076BCT027\nTHA076BCT041', fontsize=8, bbox=text_box, color='red')
plt.tight_layout()
plt.savefig('Histogram of features.png',bbox_inches='tight')
plt.show()


from sklearn.model_selection import train_test_split
X= new_df.drop(columns=['Attrition_Flag'])
Y= new_df['Attrition_Flag']
# X_train, X_test, y_train, y_test will be the resulting datasets

# Split the dataset into training and test sets (80% training and 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


from sklearn.naive_bayes import GaussianNB

# Create an instance of the Gaussian Naive Bayes model
gnb = GaussianNB()

from sklearn.impute import SimpleImputer

# Create an imputer object with strategy as 'mean'
imputer = SimpleImputer(strategy='mean')

# Fit and transform the imputer on X_train
X_train = imputer.fit_transform(X_train)

# Transform X_test using the same imputer
X_test = imputer.transform(X_test)

# Train the Gaussian Naive Bayes model
gnb.fit(X_train, y_train)

# Make predictions on the test data
y_pred = gnb.predict(X_test)

print(y_pred[:10].tolist())
print(y_test[:10].tolist())
probs = gnb.predict_proba(X_test)
print(probs[:10])

from sklearn.metrics import confusion_matrix

# Assuming gnb is your trained Gaussian Naive Bayes model and X_test, y_test are your test set and corresponding labels
y_pred = gnb.predict(X_test)
confusion_mat = confusion_matrix(y_test, y_pred)
print(confusion_mat)

import numpy as np
# Normalize the confusion matrix
normalized_confusion_mat = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]
# Create a heatmap using seaborn
plt.figure(figsize=(6, 4))
sns.heatmap(normalized_confusion_mat, annot=True, fmt=".2f", cmap="Blues")

# Add labels and title
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix GNB")

plt.text(1.8, -0.2, 'THA076BCT026\nTHA076BCT027\nTHA076BCT041', fontsize=8, bbox=text_box, color='red')
plt.tight_layout()
plt.savefig('Confusion Matrix GNB.png',bbox_inches='tight')
plt.show()

from sklearn.metrics import classification_report

report=classification_report(y_test, y_pred)

# Print the classification report
print(report)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train3 = scaler.fit_transform(X_train)
X_test3=scaler.transform(X_test)
gnb3 = GaussianNB()
gnb3.fit(X_train3, y_train)
y_pred3 = gnb3.predict(X_test3)

report=classification_report(y_test, y_pred3)
print(report)

print(numeric_df.head())
print(numeric_df.describe())

# Binning settings
age_bins = [0, 11, 21, 31, 41, 51, 61, 71, 81]
age_labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80']
credit_bins = [0, 1001, 2001, 3001, 6001, 12001, 35001]
credit_labels = ['0-1000', '1001-2000', '2001-3000', '3001-6000', '6001-12000', '12001-35000']

categorical_df = numeric_df.drop(columns=['Customer_Age', 'Credit_Limit'])

# Discretize 'Age' column
categorical_df['Age_Category'] = pd.cut(numeric_df['Customer_Age'], bins=age_bins, labels=age_labels)

# Discretize 'Credit_Limit' column
categorical_df['Credit_Category'] = pd.cut(numeric_df['Credit_Limit'], bins=credit_bins, labels=credit_labels)
categorical_df.drop(columns=['Age_Group'],inplace=True)

categorical_df.head()

categorical_df=new_df.copy()
# Discretize 'Age' column using equal width method with 5 bins
categorical_df['Age_EqualWidth'] = pd.cut(new_df['Customer_Age'], bins=5,labels=False)

# Discretize 'Fare' column using equal width method with 5 bins
categorical_df['Credit_EqualWidth'] = pd.cut(new_df['Credit_Limit'], bins=5,labels=False)
# categorical_df.drop(columns=['Age_Group'],inplace=True)

categorical_df.head()

from sklearn.naive_bayes import CategoricalNB
# Initialize the Categorical Naive Bayes model
cnb = CategoricalNB()
# Access the attributes of the classifier using the get_params() method
classifier_attributes = cnb.get_params()

# Display the attributes
print(classifier_attributes)
# Drop the rows containing 'UNKNOWN' in any column
# categorical_df =categorical_df = categorical_df.drop(index=categorical_df[categorical_df.eq('UNKNOWN').any(axis=1)].index)
X1= categorical_df.drop(columns=['Attrition_Flag'])
Y1= categorical_df['Attrition_Flag']

# Split the dataset into training and test sets (80% training and 20% test)
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, Y1, test_size=0.2, random_state=42)


# In[49]:


X_train1.isnull().sum()


# In[50]:


## Create an imputer object with strategy as 'mean'
imputer = SimpleImputer(strategy='mean')

# Fit and transform the imputer on X_train
# X_train1 = imputer.fit_transform(X_train1)
print(type(X_test1))
# Transform X_test using the same imputer
# X_test1 = imputer.transform(X_test1)
print(type(X_test1))
# Train the model on the training data
print(X_train1.shape,X_test1.shape)
classifier_attributes = cnb.get_params()

# Display the attributes
print(classifier_attributes)
# Make predictions on the test data
# Train the model on the training data
cnb.fit(X_train1, y_train1)
length=len(X_test1)
print(length)
# Make predictions on the test data
(pd.concat([X_test1[:216], X_test1[217:]])).shape
new_test = pd.concat([X_test1[:216], X_test1[217:827], X_test1[828:1934], X_test1[1935:]])
new_y_test = pd.concat([y_test1[:216], y_test1[217:827], y_test1[828:1934], y_test1[1935:]])

print(len(X_test1))
y_pred1 = cnb.predict(new_test)

print(f"Accuracy Score: {accuracy_score(y_true = new_y_test, y_pred = y_pred1) * 100:.2f} %")

confusion_mat = confusion_matrix(new_y_test, y_pred1)
print(confusion_mat)

# Normalize the confusion matrix
normalized_confusion_mat = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]
# Create a heatmap using seaborn
plt.figure(figsize=(6, 4))
sns.heatmap(normalized_confusion_mat, annot=True, fmt=".2f", cmap="Blues")

# Add labels and title
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.text(1.8, -0.2, 'THA076BCT026\nTHA076BCT027\nTHA076BCT041', fontsize=8, bbox=text_box, color='red')
plt.tight_layout()
plt.savefig('Confusion Matrix CNB_eql_width.png',bbox_inches='tight')

plt.show()
report=classification_report(new_y_test, y_pred1)

# Print the classification report
print(report)
categorical_df1=new_df.copy()

categorical_df1['Age_EqualFrequency'] = pd.qcut(numeric_df['Customer_Age'], q=5, labels=False)

# Discretize 'Age' column using equal frequency method with 5 bins
categorical_df1['Credit_EqualFrequency'] = pd.qcut(numeric_df['Credit_Limit'], q=5, labels=False)
categorical_df1.head()

X2= categorical_df1.drop(columns=['Attrition_Flag'])
Y2= categorical_df1['Attrition_Flag']
# Split the dataset into training and test sets (80% training and 20% test)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, Y2, test_size=0.2, random_state=42)
# Fit and transform the imputer on X_train
X_train2 = imputer.fit_transform(X_train2)

# Transform X_test using the same imputer
X_test2 = imputer.transform(X_test2)
# Train the model on the training data
X_test2= np.concatenate([X_test2[:216], X_test2[217:827], X_test2[828:1934], X_test2[1935:]])
y_test2= np.concatenate([y_test2[:216], y_test2[217:827], y_test2[828:1934], y_test2[1935:]])
cnb.fit(X_train2, y_train2)

# Make predictions on the test data
y_pred2 = cnb.predict(X_test2)
print(y_pred2.shape)
print(y_test2.shape)
print(X_test2.shape)

confusion_mat = confusion_matrix(y_test2, y_pred2)
print(confusion_mat)

# Normalize the confusion matrix
normalized_confusion_mat = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]
# Create a heatmap using seaborn
plt.figure(figsize=(6, 4))
sns.heatmap(normalized_confusion_mat, annot=True, fmt=".2f", cmap="Blues")

# Add labels and title
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.text(1.8, -0.2, 'THA076BCT026\nTHA076BCT027\nTHA076BCT041', fontsize=8, bbox=text_box, color='red')
plt.tight_layout()
plt.savefig('Confusion Matrix GNB_eql_freq.png',bbox_inches='tight')

plt.show()
report=classification_report(y_test2, y_pred2)

# Print the classification report
print(report)