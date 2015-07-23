##### Part 1 ##### 
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import statsmodels.formula.api as smf

# visualization
import seaborn as sns
import matplotlib.pyplot as plt


# 1. read in the yelp dataset
import pandas as pd
yelp = pd.read_csv('../hw/optional/yelp.csv', index_col = 'review_id')
yelp.head()

# 2. Perform a linear regression using 
# "stars" as your response and 
# "cool", "useful", and "funny" as predictors

'''visualizing the data'''
sns.pairplot(yelp, x_vars=['cool','useful','funny'], y_vars='stars', size=4.5, aspect=0.7)
sns.pairplot(yelp, x_vars=['cool','useful','funny'], y_vars='stars', size=4.5, aspect=0.7, kind='reg')

sns.pairplot(yelp)

yelp.corr()

sns.heatmap(yelp.corr())

'''linear regression'''
feature_cols = ['cool', 'useful', 'funny']
X = yelp[feature_cols]
y = yelp.stars

linreg = LinearRegression()
linreg.fit(X, y)

print linreg.intercept_
print linreg.coef_

zip(feature_cols, linreg.coef_)

# 3. Show your MAE, R_Squared and RMSE
'''R_squared'''
lm = smf.ols(formula='stars ~ cool + useful + funny', data=yelp).fit()
lm.rsquared
'''MAE'''
y_pred = linreg.predict(X)
print metrics.mean_absolute_error(y, y_pred)
'''RMSE'''
print np.sqrt(metrics.mean_squared_error(y, y_pred))

# 4. Use statsmodels to show your pvalues
# for each of the three predictors
# Using a .05 confidence level, 
# Should we eliminate any of the three?
lm = smf.ols(formula='stars ~ cool + useful + funny', data=yelp).fit()
lm.pvalues
lm.conf_int()
'''We believe all features have a relationship with stars rated.'''

# 5. Create a new column called "good_rating"
# this could column should be True iff stars is 4 or 5
# and False iff stars is below 4
yelp['good_rating'] = yelp['stars'] >= 4
yelp.head()
    
# 6. Perform a Logistic Regression using 
# "good_rating" as your response and the same
# three predictors
from sklearn.linear_model import LogisticRegression
Z = yelp.good_rating

logreg = LogisticRegression()
logreg.fit(X, Z)

print logreg.intercept_
print logreg.coef_

zip(feature_cols, logreg.coef_)


# 7. Show your Accuracy, Sensitivity, Specificity
# and Confusion Matrix
from sklearn import metrics
preds = logreg.predict(X)
print metrics.confusion_matrix(Z, preds)

Accuracy = (227.0 + 6733) / 10000
Sensitivity = 6733 / (130.0 + 6733)
Specificity = 227 / (227 + 2910.0)
# 8. Perform one NEW operation of your 
# choosing to try to boost your metrics!

train_test_rmse(X, y)



##### Part 2 ######

# 1. Read in the titanic data set.
titanic = pd.read_csv('titanic.csv', index_col = 'PassengerId')
titanic.head()

# 4. Create a new column called "wife" that is True
# if the name of the person contains Mrs.
# AND their SibSp is at least 1
titanic['wife'] = [row for row in titanic['Name'] if 'Mrs.' in row] and titanic['SibSp'] >= 1

# 5. What is the average age of a male and
# the average age of a female on board?
male_avg_age = titanic.Age[titanic.Sex == 'male'].mean()
female_avg_age = titanic.Age[titanic.Sex == 'female'].mean()

# 5. Fill in missing MALE age values with the
# average age of the remaining MALE ages
titanic.Age[titanic.Sex == 'male'].isnull().sum()
#titanic.Age[titanic.Sex == 'male'].fillna(value=male_avg_age,inplace=True)
titanic.Age = titanic.groupby("Sex").transform(lambda x: x.fillna(x.mean()))['Age']
'''code provided by Patrick fills in both missing male and female cells'''

# 6. Fill in missing FEMALE age values with the
# average age of the remaining FEMALE ages
titanic.Age[titanic.Sex == 'female'].isnull().sum()
#titanic.Age[titanic.Sex == 'female'].fillna(female_avg_age, inplace=True)

# 7. Perform a Logistic Regression using
# Survived as your response and age, wife
# as predictors
logreg = LogisticRegression()
titanicfeature_cols = ['Age' , 'wife']
A = titanic[titanicfeature_cols]
b = titanic.Survived
logreg.fit(A, b)
assorted_pred_class = logreg.predict(A)
# 8. Show Accuracy, Sensitivity, Specificity and 
# Confusion matrix
prds = logreg.predict(A)
print metrics.confusion_matrix(b, prds)

Accuracy = (523+26.0)/ 891
Sensitivity = (26.0) / (316+26)
Specificity = (523.0) / (523+26)


# 9. now use ANY of your variables as predictors
# Still using survived as a response to boost metrics!
cfeature_cols = ['Age','Pclass','Fare']

C = titanic[cfeature_cols]
logreg.fit(C, b)
assorted_pred_class = logreg.predict(C)

# 10. Show Accuracy, Sensitivity, Specificity
preds1 = logreg.predict(C)
print metrics.confusion_matrix(b, preds1)

Accuracy = (477+148.0)/ 891
Sensitivity = (148.0) / (194+148)
Specificity = (477.0) / (477+72)

train_test_rmse(C, b)


# REMEMBER TO USE
# TRAIN TEST SPLIT AND CROSS VALIDATION
# FOR ALL METRIC EVALUATION!!!!

