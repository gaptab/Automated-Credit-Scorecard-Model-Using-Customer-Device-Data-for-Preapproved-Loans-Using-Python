import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Dummy Data: Customer Device Usage & Loan History
np.random.seed(42)
data = {
    "Customer_ID": range(1, 501),
    "Avg_Screen_Time_Hours": np.random.uniform(2, 8, 500),  # Avg mobile usage per day
    "Browsing_Activity_Score": np.random.uniform(200, 800, 500),  # Engagement metric
    "Location_Stability": np.random.randint(0, 2, 500),  # 1: Stable, 0: Frequent moves
    "Past_Loan_Default": np.random.randint(0, 2, 500),  # 1: Defaulted, 0: Never defaulted
    "Credit_Score": np.random.randint(300, 850, 500),  # Simulated credit scores
}

df = pd.DataFrame(data)

# Target Variable: Loan Approval (1: Approved, 0: Rejected)
df["Loan_Approved"] = np.where(
    (df["Credit_Score"] > 650) & (df["Past_Loan_Default"] == 0) & (df["Location_Stability"] == 1), 1, 0
)

# Splitting Data for Model Training
X = df.drop(columns=["Customer_ID", "Loan_Approved"])
y = df["Loan_Approved"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression Model for Loan Decisioning
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Visualizing Loan Approval Rates
loan_approval_rates = df["Loan_Approved"].value_counts(normalize=True) * 100
plt.figure(figsize=(6, 4))
plt.bar(["Rejected", "Approved"], loan_approval_rates, color=["red", "green"])
plt.xlabel("Loan Decision")
plt.ylabel("Percentage of Customers")
plt.title("Loan Approval vs Rejection Rates")
plt.show()
