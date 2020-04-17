import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz


'''

Score: 0.73684

'''

# True - testing against train set to measure accuracy.
# False - testing against test set and save for submission.
is_accuracy = True

# Load data.
train_data = pd.read_csv('train.csv')

# Survived - the predicted y-value.
y = train_data.Survived

# Pclass - 1st class, 2nd class, 3rd class.
# Sex - Female 0 Male 1. Concatenated later.
# Age - null values averaged from title in Name.
# FamSize - number of family aboard the ship.
# Fare - fare value paid by passenger.
titanic_features = ['Pclass', 'Age', 'Fare', 'FamSize']

# Fill missing values in age with Title group median.
def fill_age(data_frame):
    
    # Extract titles from Name, e.g. Braund, Mr. Owen Harris.
    data_frame['Title'] = data_frame.Name.apply(
        lambda name : name.split(',')[1].split('.')[0].strip())
    
    # Normalize titles into groups.
    title_groups = {
        'Master' : 'Master',            # Master - young male.
        'Miss' : 'Miss',                # Miss - young female.
        'Mlle' : 'Miss',
        'Ms' : 'Miss',
        'Mr' : 'Mister',                # Mister - adult male.
        'Mme' : 'Mistress',             # Mistress - adult female.
        'Mrs' : 'Mistress',
        'Capt' : 'Officer',             # Officer - title attained through
        'Col' : 'Officer',              # adult institution. Gender aspecific.
        'Dr' : 'Officer',
        'Major' : 'Officer',
        'Rev' : 'Officer',
        'the Countess' : 'Royalty',     # Royalty - title fixed throughout life.
        'Don' : 'Royalty',
        'Dona' : 'Royalty',
        'Jonkheer' : 'Royalty',
        'Lady' : 'Royalty',
        'Sir' : 'Royalty'
        }
    
    # Map normalized titles to present titles.
    data_frame.Title = data_frame.Title.map(title_groups)
    
    # Group and apply median Age by each group.
    grouped = data_frame.groupby(['Title'])
    
    return grouped.Age.apply(lambda x : x.fillna(x.median()))

def get_dtc_accuracy(train_X, val_X, train_y, val_y):
    
    # Create and fit model.
    dtc_model = DecisionTreeClassifier(random_state=1)
    dtc_model.fit(train_X, train_y)
    
    # Make validation predictions.
    val_pred = dtc_model.predict(val_X)
    
    # Calculate mean absolute error.
    val_mae = mean_absolute_error(val_pred, val_y)
    print("Validation MAE when not specifying max_leaf_nodes: %s." %
          str(val_mae))

# Using best value for max leaf nodes.
def get_dtc_max_accuracy(train_X, val_X, train_y, val_y):
    
    dtc_max_model = DecisionTreeClassifier(max_leaf_nodes=100, random_state=1)
    dtc_max_model.fit(train_X, train_y)
    
    val_pred = dtc_max_model.predict(val_X)
    
    val_mae = mean_absolute_error(val_pred, val_y)
    print("Validation MAE for best value of max_leaf_nodes: %s." %
          str(val_mae))

def get_rfc_accuracy(train_X, val_X, train_y, val_y):
    
    # Create and fit model.
    rfc_model = RandomForestClassifier(random_state=1)
    rfc_model.fit(X_train, y_train)
    
    # Create predicted y values.
    val_pred = rfc_model.predict(val_X)

    val_mae = mean_absolute_error(val_pred, val_y)
    print("Validation MAE for Random Forest Model: %s." %
          str(val_mae))
    
    # Export as dot file.
    export_graphviz(rfc_model.estimators_[5], out_file='tree.dot',
                    rounded=True, proportion=False,
                    precision=2, filled=True)
    
    # Convert to png.
    from subprocess import check_call
    check_call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png'], shell=True)

# Fill missing Age values in train_data.
train_data.Age = fill_age(train_data)

# Create Sex dummy column.
sex_dummy = pd.get_dummies(train_data.Sex)

# Create new FamSize column.
train_data['FamSize'] = train_data['SibSp'] + train_data['Parch']

# Choose fit x-target.
X = pd.concat([train_data[titanic_features], sex_dummy], axis=1)

if is_accuracy:
    
    # Split into training and testing sets.
    X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                        train_size=0.5,
                                                        random_state=1)
    
    # Calculate accuracy score for each model.
    get_dtc_accuracy(X_train, X_val, y_train, y_val)
    get_dtc_max_accuracy(X_train, X_val, y_train, y_val)    
    get_rfc_accuracy(X_train, X_val, y_train, y_val)
    
else:
    
    test_data = pd.read_csv('test.csv')
    
    # Fill missing Age values in test_data.
    test_data.Age = fill_age(test_data)

    # Fill the single null value in Fare.
    test_data.Fare = test_data.Fare.fillna(0)
    
    # Create test dummy column.
    test_dummy = pd.get_dummies(test_data.Sex)
    
    # Create new FamSize column.
    test_data['FamSize'] = test_data['SibSp'] + test_data['Parch']
    
    # Choose fit x-target.
    X_test = pd.concat([test_data[titanic_features], test_dummy], axis=1)

    # Create and fit model.
    model = RandomForestClassifier(random_state=1)
    model.fit(X, y)
    
    # Create predicted y values.
    y_pred = model.predict(X_test)

    output = pd.DataFrame({'PassengerId': test_data.PassengerId,
                           'Survived': y_pred})
    output.to_csv('my_submission.csv', index=False)
    print("Your submission was successfully saved!")