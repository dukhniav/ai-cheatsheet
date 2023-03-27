# ai-cheatsheet

Basic hierchy:
- Machine Learning (ML)
  - Supervised Learning
    - Linear Regression
    - Logistic Regression
    - Decision Trees
    - Support Vector Machines
    - K-Nearest Neighbors
  - Unsupervised Learning
    - Clustering
      - K-Means
      - Hierarchical Clustering
      - DBSCAN
    - Dimensionality Reduction
      - Principal Component Analysis
      - t-SNE
      - Autoencoders
  - Semi-Supervised Learning
    - Generative Adversarial Networks
    - Self-Training
- Deep Learning (DL)
  - Deep Neural Networks
  - Convolutional Neural Networks
  - Recurrent Neural Networks
  - Transformer Networks
- Reinforcement Learning (RL)
  - Q-Learning
  - SARSA
  - Policy Gradient Methods

# Supervised vs. Unsupervised vs. Reinforcement Learning
The easiest way to distinguish a supervised learning and unsupervised learning is to see whether the data is labelled or not.

**Supervised learning** learns a function to make prediction of a defined label based on the input data. It can be either classifying data into a category (classification problem) or forecasting an outcome (regression algorithms).

**Unsupervised learning** reveals the underlying pattern in the dataset that are not explicitly presented, which can discover the similarity of data points (clustering algorithms) or uncover hidden relationships of variables (association rule algorithms) …

**Reinforcement learning** is another type of machine learning, where the agents learn to take actions based on its interaction with the environment, with the aim to maximize rewards. It is most similar to the learning process of human, following a trial-and-error approach.

# Linear Regression
At its core, linear regression is a technique for finding the best-fit line through a set of data points. The line is defined by an equation of the form: $y=mx+b$, where:
- $y$ is the output variable (also called the response variable)
- $x$ is the input variable (also called the predictor or feature variable)
- $m$ is the slope of the line
- $b$ is the y-intercept.

The goal of linear regression is to find the values of $m$ and $b$ that minimize the difference between the predicted values of $y$ and the actual values of $y$. This difference is called the **error**, and the goal is to find the line that produces the smallest error.

To find the line of best fit, linear regression uses a process called **optimization**, specifically a method called **least squares**. Least squares is a mathematical technique for minimizing the sum of the squared differences between the predicted values and the actual values.

The optimization process involves finding the values of $m$ and $b$ that minimize the sum of the squared errors. This is typically done using a method called **gradient descent**, which is an iterative process that adjusts the values of $m$ and $b$ until the sum of the squared errors is minimized.

## The data
The input and output data points represent the relationship between the dependent and independent variables that we are trying to model. In the case of linear regression, both the input and output variables are a set of continuous numeric values:

$x = [1, 2, 3, 4, 5]$ <br>
$y = [2, 4, 6, 8, 10]$

To collect the input and output data points, we typically start with a research question or hypothesis that we want to test. For example, we might be interested in understanding the relationship between a person's age and their income. In this case, we would collect data on the age and income of a sample of people.

In the above example, $x$ represents the input variable (in this case, the x-axis values on a scatterplot), and $y$ represents the output variable (in this case, the corresponding y-axis values on the same scatterplot). The goal of linear regression is to find the line of best fit that describes the relationship between $x$ and $y$.

## The model
In the case of linear regression, the model is a straight line.

## Determining the best-fit line
Linear regression uses the *least squares* method to find the values of $m$ and $b$ that minimize the sum of the squared errors.

## Evaluating the model
This can be done using metrics such as R-squared or mean squared error.

Here's an example of how to implement the least squares method for linear regression:
```python3 
import numpy as np

# Define the input and output data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

# Calculate the slope and y-intercept using the least squares method
m = ((np.mean(x) * np.mean(y)) - np.mean(x*y)) / ((np.mean(x)**2) - np.mean(x**2))
b = np.mean(y) - m*np.mean(x)

# Print the slope and y-intercept
print("Slope: ", m)
print("Y-intercept: ", b)
```

In this example, we calculate the slope and y-intercept of the line of best fit using the least squares method. We then print out the values of $m$ and $b$. Note that this is a simple example and in practice, we would typically use a library such as scikit-learn to perform linear regression.

## The assumptions
The first step is to train the algorithm on known or labeled datasets and then use the algorithm to predict unknown values. Real-life data is more complicated than the previous example. That is why linear regression analysis must mathematically modify or transform the data values to meet the following four assumptions.

### Linear relationship

A linear relationship must exist between the independent and dependent variables. To determine this relationship, data scientists create a scatter plot—a random collection of x and y values—to see whether they fall along a straight line. If not, you can apply nonlinear functions such as square root or log to mathematically create the linear relationship between the two variables.

### Residual independence

Data scientists use residuals to measure prediction accuracy. A residual is the difference between the observed data and the predicted value. Residuals must not have an identifiable pattern between them. For example, you don't want the residuals to grow larger with time. You can use different mathematical tests, like the Durbin-Watson test, to determine residual independence. You can use dummy data to replace any data variation, such as seasonal data.

### Normality

Graphing techniques like Q-Q plots determine whether the residuals are normally distributed. The residuals should fall along a diagonal line in the center of the graph. If the residuals are not normalized, you can test the data for random outliers or values that are not typical. Removing the outliers or performing nonlinear transformations can fix the issue.

### Homoscedasticity

Homoscedasticity assumes that residuals have a constant variance or standard deviation from the mean for every value of x. If not, the results of the analysis might not be accurate. If this assumption is not met, you might have to change the dependent variable. Because variance occurs naturally in large datasets, it makes sense to change the scale of the dependent variable. For example, instead of using the population size to predict the number of fire stations in a city, might use population size to predict the number of fire stations per person.

## Different types of linear regression
### Simple linear regression

Simple linear regression is defined by the linear function:
$Y= β0*X + β1 + ε $

$β0$ and $β1$ are two unknown constants representing the regression slope, whereas $ε$ (epsilon) is the error term.

You can use simple linear regression to model the relationship between two variables, such as these:
- Rainfall and crop yield
- Age and height in children
- Temperature and expansion of the metal mercury in a thermometer

#### Multiple linear regression

In multiple linear regression analysis, the dataset contains one dependent variable and multiple independent variables. The linear regression line function changes to include more factors as follows:

$Y= β0*X0 + β1X1 + β2X2+…… βnXn+ ε $

As the number of predictor variables increases, the $β$ constants also increase correspondingly.

Multiple linear regression models multiple variables and their impact on an outcome:
- Rainfall, temperature, and fertilizer use on crop yield
- Diet and exercise on heart disease
- Wage growth and inflation on home loan rates

#### Logistic regression

Data scientists use logistic regression to measure the probability of an event occurring. The prediction is a value between 0 and 1, where 0 indicates an event that is unlikely to happen, and 1 indicates a maximum likelihood that it will happen. Logistic equations use logarithmic functions to compute the regression line.

These are some examples:
- The probability of a win or loss in a sporting match
- The probability of passing or failing a test 
- The probability of an image being a fruit or an animal

# Logistic Regression

Logistic regression is a statistical method used to model the relationship between a set of input variables and a binary output variable (i.e., a variable that takes on one of two possible values). It is a type of supervised learning algorithm that falls under the broader category of machine learning.

The goal of logistic regression is to predict the probability of the binary output variable taking on a certain value, given the values of the input variables. The output of the logistic regression model is a probability value between 0 and 1, which can be converted into a binary classification using a decision threshold.

## How Logistic Regression Works

Logistic regression is a type of regression analysis, similar to linear regression. However, instead of fitting a straight line to the data, logistic regression models the relationship between the input variables and the logarithm of the odds of the binary output variable taking on a certain value.

To fit a logistic regression model, we start by collecting a set of input and output data points. The input variables can be continuous, discrete, or categorical, while the output variable is binary. We then use the collected data to estimate the model parameters, which describe the relationship between the input variables and the output variable.

The logistic regression model uses a mathematical function called the logistic function (also known as the sigmoid function) to model the probability of the binary output variable taking on a certain value. The logistic function is defined as follows:

$σ(z) = 1 / (1 + e^-z)$

where z is a linear combination of the input variables and their corresponding model parameters:

$z = β0 + β1x1 + β2x2 + ... + βpxp$

Here, x1, x2, ..., xp represent the input variables, and β0, β1, β2, ..., βp represent the model parameters. The goal of logistic regression is to estimate the values of the model parameters that best describe the relationship between the input variables and the output variable.

## Estimating Model Parameters

To estimate the values of the model parameters in logistic regression, we use a process called maximum likelihood estimation. This involves finding the values of the model parameters that maximize the likelihood function, which measures the probability of the observed data given the model parameters.

The likelihood function is defined as follows:

$L(β) = ∏(i=1)^n σ(z_i)^yi * (1 - σ(z_i))^(1 - yi)$

where n is the number of observations in the dataset, yi is the binary output variable for observation i, and zi is the linear combination of input variables and model parameters for observation i.

Maximizing the likelihood function involves finding the values of the model parameters that make the observed data most probable. This is typically done using optimization algorithms, such as gradient descent.

## Interpreting Model Parameters

Once we've estimated the values of the model parameters using maximum likelihood estimation, we can use them to interpret the relationship between the input variables and the binary output variable. Each model parameter represents the change in the log-odds of the binary output variable for a one-unit change in the corresponding input variable, holding all other variables constant.

For example, suppose we have a logistic regression model with two input variables x1 and x2, and corresponding model parameters β1 and β2. If we hold x2 constant and increase x1 by one unit, the log-odds of the binary output variable will increase by β1, assuming all other variables are held constant.

## Regularization

In some cases, logistic regression models may suffer from overfitting, where the model is too complex and captures noise in the data rather than the underlying relationship between the input and output variables. To address this, we can use regularization techniques, which add a penalty term to the likelihood function that discourages large values of the model parameters.

Two common regularization techniques used in logistic regression are L1 regularization (also known as Lasso) and L2 regularization (also known as Ridge). L1 regularization adds a penalty term equal to the absolute value of the model parameters, while L2 regularization adds a penalty term equal to the squared value of the model parameters.

## Simple Project: Predicting Diabetes

To solidify your understanding of logistic regression, you could try building a model to predict whether a patient has diabetes based on their medical information. You could use the Pima Indians Diabetes Database, which contains medical information for female patients of Pima Indian heritage, and a binary indicator variable for whether or not they developed diabetes within five years.

Here's some sample code in Python using the scikit-learn library to fit a logistic regression model to the dataset:

```python3
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv'
dataset = pd.read_csv(url, header=None)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit logistic regression model
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Make predictions on test set
y_pred = lr.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f%%' % (accuracy * 100.0))
```

In this code, we first load the dataset from a URL and split it into training and testing sets using train_test_split. We then fit a logistic regression model using LogisticRegression, and make predictions on the test set using predict. Finally, we evaluate the model performance using accuracy_score.

## Multiclass Logistic Regression

So far, we've only discussed binary logistic regression, where the output variable can take on two possible values. However, logistic regression can be extended to handle multiclass classification problems, where the output variable can take on more than two possible values.

One way to do this is to use a technique called one-vs-all (OVA) or one-vs-rest (OVR) classification. In this approach, we train k separate binary logistic regression models, where k is the number of classes in the problem. For each model, we treat one class as the "positive" class and all other classes as the "negative" class. To make a prediction for a new input, we compute the probability of belonging to each class using the corresponding logistic regression model, and choose the class with the highest probability.

## Imbalanced Classes

In some classification problems, the distribution of classes in the training data may be highly imbalanced, where one class occurs much less frequently than the other. This can make it difficult for the logistic regression model to learn the relationship between the input and output variables for the minority class.

To address this, we can use techniques such as oversampling the minority class, undersampling the majority class, or using cost-sensitive learning, where we assign different misclassification costs to the different classes.

## Simple Project: Predicting Titanic Survivors

Another way to solidify your understanding of logistic regression is to build a model to predict the likelihood of surviving the sinking of the Titanic, based on passenger data. You can use the Titanic dataset, which contains information on passengers including their age, sex, passenger class, and whether or not they survived.

Here's some sample code in Python using the pandas and scikit-learn libraries to fit a logistic regression model to the dataset:

```python3
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
dataset = pd.read_csv(url)
X = dataset[['Age', 'Sex', 'Pclass']]
y = dataset['Survived']

# Preprocess data
X['Sex'] = X['Sex'].apply(lambda x: 1 if x == 'male' else 0)
X = pd.get_dummies(X, columns=['Pclass'], prefix='Pclass')

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit logistic regression model
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Make predictions on test set
y_pred = lr.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f%%' % (accuracy * 100.0))
```

In this code, we first load the dataset from a URL and preprocess it by converting the Sex variable to a binary indicator variable and creating indicator variables for the Pclass variable using get_dummies. We then split the data into training and testing sets using train_test_split, fit a logistic regression model using LogisticRegression, and make predictions on the test set using predict. Finally, we evaluate the model performance using accuracy_score.