
# AOlearn: An AO Machine Learning Package

AOlearn is a APM package that provides implementations of common machine learning algorithms for AO Hyper Parallel Computer. It aims to make machine learning accessible and easy to use within Lua projects.

## Submodules

AOlearn is organized into submodules, each focusing on a specific category of algorithms or tasks. The current submodules include:

- `AOlearn.linear_regression`: Provides linear regression models.
- `AOlearn.logistic`: Implements logistic regression for binary classification.
- `AOlearn.lasso`: Includes Lasso regression for feature selection and regularization.
- `AOlearn.ridge`: Provides Ridge regression for regularization.

## Submodule Documentation

### `AOlearn.linear_regression`

This submodule provides functions for fitting and predicting with linear regression models.

#### Functions:

- `fit_linear(X, Y)`:
  - Fits a linear regression model to the provided data.
  - **Parameters:**
    - `X`: A table of arrays, where each array represents a feature.
    - `Y`: An array containing the target values.
  - **Returns:**
    - `gains`: A table containing the learned coefficients (slopes) for each feature.
    - `offset`: The learned intercept (bias).
- `predict_linear(gains, offset, inputFeatures)`:
  - Predicts the target value for a given data point using the fitted linear model.
  - **Parameters:**
    - `gains`: The learned coefficients (slopes) for each feature.
    - `offset`: The learned intercept (bias).
    - `inputFeatures`: A table containing the feature values for the data point.
  - **Returns:**
    - The predicted target value.

### `AOlearn.logistic`

This submodule provides functionality for performing binary classification using Logistic Regression.

#### Functions:
- `fit_logistic(X, y, learningRate, numIterations)`:
  - Fits a logistic regression model to the given data using gradient descent.
  - **Parameters:**
    - `X`: A table of arrays, where each array represents a data point's features.
    - `y`: An array containing the target labels (0 or 1) for each data point.
    - `learningRate`: (Optional) The learning rate for gradient descent. Defaults to 0.01.
    - `numIterations`: (Optional) The number of iterations to run gradient descent. Defaults to 1000.
  - **Returns:**
    - `weights`: A table containing the learned coefficients for each feature.
    - `bias`: The learned intercept term (bias).
- `predict_logistic(weights, bias, features)`:
  - Predicts the probability of the positive class (class 1) for a given data point using the learned model.
  - **Parameters:**
    - `weights`: The learned coefficients for each feature (returned by `fit_logistic`).
    - `bias`: The intercept term (bias) (returned by `fit_logistic`).
    - `features`: A table containing the feature values for the data point to predict.
  - **Returns:**
    - The predicted probability of the positive class (between 0 and 1).

### `AOlearn.lasso`

This submodule implements Lasso regression, a linear regression model that incorporates L1 regularization.

#### Functions:
- `fit_lasso(X, Y, lambda, learningRate, numIterations)`:
  - Fits a Lasso regression model to the given data using gradient descent.
  - **Parameters:**
    - `X`: A table of arrays, where each array represents a feature.
    - `Y`: An array containing the target values.
    - `lambda`: The regularization parameter (controls the strength of regularization).
    - `learningRate`: (Optional) The learning rate for gradient descent.
    - `numIterations`: (Optional) The number of iterations for gradient descent.
  - **Returns:** 
    - `weights`: A table containing the learned coefficients for each feature.
    - `bias`: The learned intercept term (bias).
- `predict_lasso(weights, bias, inputFeatures)`:
  - Predicts the target value for a given data point using the fitted Lasso model. This function is the same as `predict_linear` in the `AOlearn.linear_regression` submodule.

### `AOlearn.ridge`

This submodule implements Ridge regression, a linear regression model that incorporates L2 regularization.

#### Functions:
- `fit_ridge(X, Y, lambda, learningRate, numIterations)`:
  - Fits a Ridge regression model to the given data using gradient descent.
  - **Parameters:**
    - `X`: A table of arrays, where each array represents a feature.
    - `Y`: An array containing the target values.
    - `lambda`: The regularization parameter (controls the strength of regularization).
    - `learningRate`: (Optional) The learning rate for gradient descent.
    - `numIterations`: (Optional) The number of iterations for gradient descent.
  - **Returns:** 
    - `weights`: A table containing the learned coefficients for each feature.
    - `bias`: The learned intercept term (bias).
- `predict_ridge(weights, bias, inputFeatures)`:
  - Predicts the target value for a given data point using the fitted Ridge model. This function is the same as `predict_linear` in the `AOlearn.linear_regression` submodule.

## Example Usage

```lua
-- Example 1: Linear Regression
-- Sample data: House price prediction based on size and number of bedrooms
local house_features = {
  {1500,2000,1200,2500},  -- Size (sqft), Bedrooms
  {3,4,2,5},

}
local house_prices = {250000, 350000, 180000, 420000}

-- Fit a linear regression model
local gains, offset = AOlearn.linear_regression.fit_linear(house_features, house_prices)

-- Predict the price of a new house with 1800 sqft and 3 bedrooms
local new_house_features = {1800, 3}
local predicted_price = AOlearn.linear_regression.predict_linear(gains, offset, new_house_features)

print("Predicted House Price:", predicted_price)
```
```lua

-- Example 2: Logistic Regression

-- Sample data: Spam classification based on email features (simplified)
local email_features = {
  {1, 0, 5}, -- Feature 1, Feature 2, Feature 3
  {0, 1, 2},
  {1, 1, 8},
  {0, 0, 1}
}
local is_spam = {1, 0, 1, 0} -- 1: Spam, 0: Not spam

-- Fit a logistic regression model
local weights, bias = AOlearn.logistic.fit_logistic(email_features, is_spam)

-- Predict the probability of spam for a new email
local new_email_features = {0, 1, 6}
local spam_probability = AOlearn.logistic.predict_logistic(weights, bias, new_email_features)

print("Spam Probability:", spam_probability)

if spam_probability > 0.5 then 
  print("This email is likely spam.")
else
  print("This email is likely not spam.")
end
```
```lua

-- Example 3: Lasso Regression (for feature selection)

-- Sample data (same as linear regression, but with potentially irrelevant features)
local house_features_lasso = {
  {1500, 3, 1},  -- Size, Bedrooms, Irrelevant Feature
  {2000, 4, 0},
  {1200, 2, 1},
  {2500, 5, 0}
}

-- Fit a Lasso regression model (lambda controls the strength of regularization)
local weights_lasso, bias_lasso = AOlearn.lasso.fit_lasso(house_features_lasso, house_prices, 0.5)

-- Observe the weights - Lasso tends to shrink irrelevant feature weights towards zero
for i, w in ipairs(weights_lasso) do
  print("Feature", i, "Weight:", w)
end

local new_house_features = {1800, 3}
local predicted_price = AOlearn.lasso.predict_lasso(weights_lasso, bias_lasso, new_house_features)

print("Predicted House Price:", predicted_price)
```
```lua

-- Example 4: ridge Regression (for feature selection)

-- Sample data (same as linear regression, but with potentially irrelevant features)
local house_features_ridge = {
  {1500, 3, 1},  -- Size, Bedrooms, Irrelevant Feature
  {2000, 4, 0},
  {1200, 2, 1},
  {2500, 5, 0}
}

-- Fit a Lasso regression model (lambda controls the strength of regularization)
local weights_ridge, bias_ridge = AOlearn.ridge.fit_ridge(house_features_lasso, house_prices, 0.5)

-- Observe the weights - Lasso tends to shrink irrelevant feature weights towards zero
for i, w in ipairs(weights_ridge) do
  print("Feature", i, "Weight:", w)
end

local new_house_features = {1800, 3}
local predicted_price = AOlearn.ridge.predict_ridge(weights_ridge, bias_ridge, new_house_features)

print("Predicted House Price:", predicted_price)
```

**Explanation:**

1. **Linear Regression:** We use linear regression to predict house prices based on size and the number of bedrooms. The `fit_linear` function learns the relationship between these features and prices, allowing us to make predictions for new houses.

2. **Logistic Regression:** We use logistic regression to classify emails as spam or not spam based on some simplified features. The model outputs a probability of an email being spam, and we can set a threshold (e.g., 0.5) to make a decision.

3. **Lasso Regression:** We demonstrate how Lasso regression can help with feature selection. By adding a regularization penalty, Lasso encourages the model to assign less importance (smaller weights) to less relevant features. In this example, we might observe that the weight for the "Irrelevant Feature" is close to zero, indicating its lack of predictive power.

**Key Points:**

- **Data Preparation:**  It's crucial to have your data in the correct format (tables of features and targets) before using AOlearn's functions.
- **Feature Engineering:** The success of your models heavily relies on selecting and engineering relevant features.
- **Model Selection:** Choose the appropriate algorithm (linear, logistic, Lasso, Ridge) based on your problem type (regression or classification) and your goals (prediction accuracy, feature selection, etc.).
- **Hyperparameter Tuning:** Experiment with different hyperparameters (like `learningRate` and `lambda`) to find the best settings for your specific data.