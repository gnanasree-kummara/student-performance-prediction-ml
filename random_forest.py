import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv("data/StudentPerformanceFactors.csv")
df.fillna(df.mode().iloc[0], inplace=True)

# ----- Grade Categorization -----
def grade(x):
    if x >= 90: return 'A'
    elif x >= 85: return 'B'
    elif x >= 75: return 'C'
    elif x >= 68: return 'D'
    else: return 'F'

df['target'] = df['Exam_Score'].apply(grade)
df.drop("Exam_Score", axis=1, inplace=True)

# Encoding
le = LabelEncoder()
df['target'] = le.fit_transform(df['target'])
for col in df.columns:
    if df[col].dtype == "object" and col != "target":
        df[col] = le.fit_transform(df[col])

X = df.drop("target", axis=1)
y = df["target"]

# Apply SMOTE
sm = SMOTE(random_state=42)
X, y = sm.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Model Training
model = RandomForestClassifier(n_estimators=250, random_state=42)
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

# Output Only the Final Result
print("\nRandom Forest Accuracy:", accuracy_score(y_test, pred))
print("\nClassification Report:\n")
print(classification_report(y_test, pred))
