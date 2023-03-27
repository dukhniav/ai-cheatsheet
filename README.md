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
