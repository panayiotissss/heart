import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

#Load data

df = pd.read_csv('data/disease_prediction.csv')
df.drop(columns=['patient_id'], inplace=True)

#Separate features and target
X, y = df.drop(columns=['disease']), df['disease']

y = y.map({'Yes': 1, 'No': 0})

#Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)#stratify=y (if inbalanced))





#Define columns
numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()


# Build preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', RobustScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
    ]
)



# Pipeline 1: Logistic Regression
pipeline_lr = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(max_iter=1000))
])

# Pipeline 2: Random Forest
pipeline_rf = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])



# Train
pipeline_lr.fit(X_train, y_train) #we use fit to train
pipeline_rf.fit(X_train, y_train)

# Predict
y_pred_lr = pipeline_lr.predict(X_test) #we use predict on test
y_pred_rf = pipeline_rf.predict(X_test)


