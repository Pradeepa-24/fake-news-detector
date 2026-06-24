import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

# Load datasets
fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

# Add labels
fake["label"] = 0
true["label"] = 1

# Combine datasets
data = pd.concat([fake, true]).sample(frac=1, random_state=42)

# Use only text column
X = data["text"]
y = data["label"]

# Convert text to numbers
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

print("\nTesting with actual dataset examples:")

real_sample = true["text"].iloc[0]
fake_sample = fake["text"].iloc[0]

print("Real sample prediction:",
      model.predict(vectorizer.transform([real_sample])))

print("Fake sample prediction:",
      model.predict(vectorizer.transform([fake_sample])))

sample = ["The government announced a new education policy to improve digital learning in schools across India."]

sample_vec = vectorizer.transform(sample)

print("Sample Prediction:", model.predict(sample_vec))

# Save model
joblib.dump(model, "model/model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("Model saved successfully!")