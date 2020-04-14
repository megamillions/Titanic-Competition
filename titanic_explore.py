import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

'''

Score: 0.73684

'''

# True - testing against train set to measure accuracy.
# False - testing against test set to save for submission.
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
    print(data_frame.Title.value_counts())
    # Group and apply median Age by each group.
    grouped = data_frame.groupby(['Title'])
    
    sns.swarmplot(x=data_frame.Title, y=data_frame.Survived)
    plt.show()
    
    return grouped.Age.apply(lambda x : x.fillna(x.median()))

# Fill missing Age values in train_data.
train_data.Age = fill_age(train_data)

# Create Sex dummy column.
sex_dummy = pd.get_dummies(train_data.Sex)

# Create new FamSize column.
train_data['FamSize'] = train_data['SibSp'] + train_data['Parch']

# Choose fit x-target.
X = pd.concat([train_data[titanic_features], sex_dummy], axis=1)

if (is_accuracy):
    
    # Split into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            train_size=0.5,
                                                            random_state=1)
    
    # Create and fit model.
    model = RandomForestClassifier(random_state=1)
    model.fit(X_train, y_train)
    
    # Create predicted y values.
    y_pred = model.predict(X_test)
    
    # Calculate accuracy score.
    accuracy = accuracy_score(y_test, y_pred)
    
    print('Your model has a predicted accuracy score of %s' % str(accuracy))
    
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