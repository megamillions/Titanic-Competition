import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

'''

Baseline model. Kaggle score: 0.77511

'''

train_data = pd.read_csv('train.csv')

# Survived - the predicted y-value.
y = train_data.Survived

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])

'''

Needed for final. Not needed in accuracy check.

test_data = pd.read_csv('test.csv')
X_test = pd.get_dummies(test_data[features])

'''

# Split into training and testing sets to measure accuracy.
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.5,
                                                    random_state=1)

# Create and fit model.
model = RandomForestClassifier(n_estimators=100, max_depth=5,
                                random_state=1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Calculate accuracy score.
accuracy = accuracy_score(y_test, y_pred)

print('Base model accuracy score is...')
print(accuracy)