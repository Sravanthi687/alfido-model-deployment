import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

def main():
    print("Loading Iris dataset...")
    # Load the Iris dataset
    # Features: sepal length, sepal width, petal length, petal width
    # Target: species (0: setosa, 1: versicolor, 2: virginica)
    data = load_iris()
    X, y = data.data, data.target

    print("Splitting dataset into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Logistic Regression model...")
    # Train a simple Logistic Regression model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    print("Evaluating model...")
    # Evaluate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")

    print("Saving model to model.pkl...")
    # Save the trained model to disk
    # We save it in the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "model.pkl")
    
    joblib.dump(model, model_path)
    print(f"Model successfully saved to: {model_path}")

if __name__ == "__main__":
    main()
