import csv
import sys
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

TEST_SIZE = 0.4

def main():
    
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train to a model and then make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    #  results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")

def load_data(filename):
    """
    Load shopping data from a CSV file """

    # mapping months
    month_mapping = {
        "Jan": 0, "Feb": 1, "Mar": 2, "Apr": 3, "May": 4, "June": 5,
        "Jul": 6, "Aug": 7, "Sep": 8, "Oct": 9, "Nov": 10, "Dec": 11
    }

    evidence = []
    labels = []

    # Read CSV file
    with open(filename, mode="r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            evidence.append([
                int(row["Administrative"]),
                float(row["Administrative_Duration"]),
                int(row["Informational"]),
                float(row["Informational_Duration"]),
                int(row["ProductRelated"]),
                float(row["ProductRelated_Duration"]),
                float(row["BounceRates"]),
                float(row["ExitRates"]),
                float(row["PageValues"]),
                float(row["SpecialDay"]),
                month_mapping[row["Month"]],
                int(row["OperatingSystems"]),
                int(row["Browser"]),
                int(row["Region"]),
                int(row["TrafficType"]),
                1 if row["VisitorType"] == "Returning_Visitor" else 0,
                1 if row["Weekend"] == "TRUE" else 0
            ])
            labels.append(1 if row["Revenue"] == "TRUE" else 0)

    return evidence, labels

def train_model(evidence, labels):
   
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model

def evaluate(labels, predictions):
  
    labels = np.array(labels)
    predictions = np.array(predictions)

    # Sensitivity: True Positive Rate
    true_positive = ((labels == 1) & (predictions == 1)).sum()
    actual_positive = (labels == 1).sum()
    sensitivity = np.divide(true_positive, actual_positive, 
out=np.zeros(1), where=actual_positive > 0)[0]

    # Specificity: True Negative Rate
    true_negative = ((labels == 0) & (predictions == 0)).sum()
    actual_negative = (labels == 0).sum()
    specificity = np.divide(true_negative, actual_negative, 
out=np.zeros(1), where=actual_negative > 0)[0]

    return sensitivity, specificity

if __name__ == "__main__":
    main()

