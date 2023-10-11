from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample data (replace this with your actual data)
X = ["Hello, this is a legitimate email.",
     "Congratulations! You've won a million dollars!",
     "Meeting scheduled for tomorrow.",
     "Claim your prize now! Limited time offer!"]

y = [0, 1, 0, 1]  # 0 for ham, 1 for spam

# Convert text data to numerical features using CountVectorizer
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Choose a machine learning model (Naive Bayes)
model = MultinomialNB()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
