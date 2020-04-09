import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load data.
train_data = pd.read_csv('train.csv')

# Filter rows with missing values.
train_data.dropna(inplace=True)

# Choose fit y-target.
y = train_data.Survived

'''

First model is here. Score: unknown

'''

# Choose features.
titanic_features = ['Pclass', 'Age', 'Fare', 'FamSize']

# Create Sex dummy column.
sex_dummy = pd.get_dummies(train_data.Sex)

# Create new FamSize column.
train_data['FamSize'] = train_data['SibSp'] + train_data['Parch']

# Choose fit x-target.
X1 = pd.concat([train_data[titanic_features], sex_dummy], axis=1)

# Split into training and testing sets.
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y,
                                                        train_size=0.5,
                                                        random_state=1)

# Create and fit model.
model1 = RandomForestClassifier(random_state=1)
model1.fit(X1_train, y1_train)

y1_pred = model1.predict(X1_test)

# Calculate accuracy score.
accuracy1 = accuracy_score(y1_test, y1_pred)

print(y1_pred[:9])
print('First model accuracy score is...')
print(accuracy1)

plt.ylabel('y1_pred')
sns.swarmplot(x=y1_test, y=y1_pred)
plt.show()

'''

Baseline model below this point. Score: 0.77511

'''

def BaseModel():
    base_features = ["Pclass", "Sex", "SibSp", "Parch"]
    X0 = pd.get_dummies(train_data[base_features])
    
    # Split into training and testing sets.
    X0_train, X0_test, y0_train, y0_test = train_test_split(X0, y,
                                                            train_size=0.5,
                                                            random_state=1)
    
    # Create and fit model.
    model0 = RandomForestClassifier(n_estimators=100, max_depth=5,
                                    random_state=1)
    model0.fit(X0_train, y0_train)
    
    y0_pred = model0.predict(X0_test)
    
    # Calculate accuracy score.
    accuracy0 = accuracy_score(y0_test, y0_pred)
    
    print(y0_pred[:9])
    print('Base model accuracy score is...')
    print(accuracy0)
    
    plt.ylabel('y0_pred')
    sns.swarmplot(x=y0_test, y=y0_pred)
    plt.show()
    
    print(train_data.columns)
    
BaseModel()

'''

Print for submission below this line.

'''

test_data = pd.read_csv('test.csv')

# Filter rows with missing values.
test_data.dropna(inplace=True)

# Create test dummy column.
test_dummy = pd.get_dummies(test_data.Sex)

# Create new FamSize column.
test_data['FamSize'] = test_data['SibSp'] + test_data['Parch']

# Choose fit x-target.
X_test = pd.concat([test_data[titanic_features], test_dummy], axis=1)

# Use first model, pre-split, to fit test.
yP_pred = model1.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId,
                       'Survived': yP_pred})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")