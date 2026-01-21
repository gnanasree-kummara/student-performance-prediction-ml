import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

df = pd.read_csv("data/StudentPerformanceFactors.csv")
df.fillna(df.mode().iloc[0], inplace=True)

# Create Grade Labels
def grade(x):
    if x >= 85: return 'A'
    elif x >= 75: return 'B'
    elif x >= 65: return 'C'
    elif x >= 50: return 'D'
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

print("\nBefore SMOTE:\n", y.value_counts())

sm = SMOTE(random_state=42)
X, y = sm.fit_resample(X, y)

print("\nAfter SMOTE:\n", pd.Series(y).value_counts())

# Scaling (important for SVM)
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Model
model = SVC(kernel='rbf', decision_function_shape='ovr')
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("\n⚙️ SVM Accuracy:", accuracy_score(y_test, pred))
print("\n", classification_report(y_test, pred))
