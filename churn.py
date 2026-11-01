import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df = pd.DataFrame({
    'Tenure': [1, 2, 5, 10, 12, 24, 36, 48, 60, 72],
    'MonthlyCharges': [80, 90, 75, 60, 65, 50, 45, 40, 35, 30],
    'Contract': ['Month-to-month']*4 + ['One year']*3 + ['Two year']*3,
    'Churn': ['Yes']*4 + ['No']*6
})

X = df[['Tenure', 'MonthlyCharges', 'Contract']]
y = (df['Churn'] == 'Yes').astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

pre = ColumnTransformer([
    ('num', StandardScaler(), ['Tenure', 'MonthlyCharges']),
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['Contract']),
])
model = Pipeline([('pre', pre), ('clf', LogisticRegression())])
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print(classification_report(y_test, y_pred, zero_division=0))

new_customer = pd.DataFrame({
    'Tenure': [3], 'MonthlyCharges': [85], 'Contract': ['Month-to-month']
})
prob = model.predict_proba(new_customer)[0][1]
print("May leave." if prob >= 0.5 else "Likely to stay.")
print("Churn Probability:", round(prob * 100, 2), "%")
