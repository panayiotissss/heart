#Load libraries/modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Load data

df = pd.read_csv('data/disease_prediction.csv')
df.drop(columns=['patient_id'], inplace=True)

#Inspect

#print(df.columns)
#print(df.info())
#print(df.describe())



#Check distribution of target variable

print(df['disease'].value_counts())

#Check-Remove Nulls

print(df.isnull().sum())
#df.dropna(inplace=True)                        # drop any row with a null
#df.dropna(subset=['age', 'bmi'], inplace=True) #only drops if na in this columns
#df['age'].fillna(df['age'].median(), inplace=True)    # numeric — use median
#df['gender'].fillna(df['gender'].mode()[0], inplace=True)  # categorical — use mode

#Check-Remove Duplicates

print(df.duplicated().sum())
#df.drop_duplicates(inplace=True)

#Check extreme outliers (bad data, not statistical)

num_features = ['age', 'bmi', 'glucose_mg_dl', 'cholesterol_mg_dl', 'systolic_bp', 'diastolic_bp', 'heart_rate']
print(df[num_features].describe())

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for ax, feature in zip(axes.flatten(), num_features):
    ax.boxplot(df[feature].dropna())
    ax.set_title(feature)
axes[1, 3].set_visible(False)
plt.tight_layout()
plt.show()



#Check numerical features in response to target
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
features = ['age', 'bmi', 'glucose_mg_dl', 'cholesterol_mg_dl']

for ax, feature in zip(axes.flatten(), features):
      df[df['disease'] == 'No'][feature].hist(ax=ax, alpha=0.5, label='No', bins=20)
      df[df['disease'] == 'Yes'][feature].hist(ax=ax, alpha=0.5, label='Yes', bins=20)
      ax.set_title(feature)
      ax.legend()

plt.tight_layout()
plt.show()


#Check categorical features in response to target
cat_features = ['gender', 'smoking', 'alcohol_consumption', 'physical_activity', 'family_history']                                                                                                       
fig, axes = plt.subplots(2, 3, figsize=(15, 8))                                                                                                                                                          
                                                                                                                                                                                                         
for ax, feature in zip(axes.flatten(), cat_features):                                                                                                                                                    
    rates = (df.groupby(feature)['disease']                                                                                                                                                              
                     .value_counts(normalize=True)                                                                                                                                                             
                     .rename('proportion')                                                                                                                                                                     
                     .reset_index())                                                                                                                                                                           
    sns.barplot(data=rates, x=feature, y='proportion', hue='disease', ax=ax)                                                                                                                             
    ax.set_title(feature)                                                                                                                                                                                
    ax.set_ylabel('proportion')                                                                                                                                                                          
                                                                                                                                                                                                               
axes[1, 2].set_visible(False)                                                                                                                                                                            
plt.tight_layout()                                                                                                                                                                                       
plt.show()                                                                                                                                                                                               
               




# Correlation Matrix (Linear relationships)

numeric_df = df.select_dtypes(include='number')  # drops categoricals automatically
numeric_df['disease'] = df['disease'].map({'Yes': 1, 'No': 0})
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()  