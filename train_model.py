# train_model.py
import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model and test data
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('test_data.pkl', 'wb') as f:
    pickle.dump((X_test, y_test), f)