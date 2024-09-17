
local function CalculateLinearRegression(X, Y)
    -- Input:
    --   X: A table of arrays, where each array represents a feature (e.g., X[1] is x1, X[2] is x2, etc.)
    --   Y: An array containing the target values (y)
    -- Output: A table of gains (slopes) for each feature and the offset (y-intercept)

    -- Check if inputs are valid
      if not (X and Y and # X > 0 and # Y == # X[1]) then
          error("Invalid input: X should be a table of arrays, and Y should be an array of equal length to each array in X.")
      end
      local numFeatures = # X
      local numDataPoints = # Y

    -- Create a table to store sums for each feature and y
      local sums = {}
      for j = 1, numFeatures do
          sums[j] = 0
      end
      sums[numFeatures + 1] = 0  -- Extra entry for sum of Y
      local sumProducts = {} -- Table to store sum of products (x1*y, x2*y, etc.)
      for j = 1, numFeatures do
          sumProducts[j] = 0
      end
      local sumSquares = {} -- Table to store sum of squares for each feature
      for j = 1, numFeatures do
          sumSquares[j] = 0
      end

    -- Calculate the sums
      for i = 1, numDataPoints do
          for j = 1, numFeatures do
              sums[j] = sums[j] + X[j][i]
              sumProducts[j] = sumProducts[j] + X[j][i] * Y[i]
              sumSquares[j] = sumSquares[j] + X[j][i] * X[j][i]
          end
          sums[numFeatures + 1] = sums[numFeatures + 1] + Y[i]
      end

    -- Calculate the gains (slopes) for each feature
      local gains = {}
      for j = 1, numFeatures do
          gains[j] = (numDataPoints * sumProducts[j] - sums[j] * sums[numFeatures + 1]) / (numDataPoints * sumSquares[j] - sums[j] * sums[j])
      end

    -- Calculate the offset (y-intercept)
      local offset = (sums[numFeatures + 1] - (gains[1] * sums[1])) / numDataPoints -- Using x1 for offset calculation, you can adjust if needed
      return gains, offset
  end

  local function CalculateResult(gains, offset, inputFeatures)
    -- Input:
    --   gains: A table containing the gains (slopes) for each feature
    --   offset: The offset (y-intercept)
    --   inputFeatures: A table containing the input values for each feature
    --                  (e.g., {2, 5, 1} for x1=2, x2=5, x3=1)
    -- Output:
    --   The calculated result (predicted y value)

    -- Check if inputs are valid
      if not (gains and offset and inputFeatures and # gains == # inputFeatures) then
          error("Invalid input: Ensure gains, offset, and inputFeatures are provided correctly.")
      end
      local result = offset
      for i = 1, # gains do
          result = result + gains[i] * inputFeatures[i]
      end
      return result
  end

  -- Helper Function: Sigmoid for Logistic Regression
  local function sigmoid(z)
      return 1 / (1 + math.exp(- z))
  end

  -- Helper Function: Sign for Lasso/Ridge Regression
  function math.sign(x)
      if x > 0 then
          return 1
      elseif x < 0 then
          return -1
      else
          return 0
      end
  end

  -- Logistic Regression with Gradient Descent
  local function logisticRegression(X, y, learningRate, numIterations)
      local numFeatures = # X[1]
      local numSamples = # X
      local weights = {}
      for i = 1, numFeatures do
          weights[i] = 0
      end  -- Initialize weights to 0
      local bias = 0
      for iteration = 1, numIterations do
          local dw = {}
          for i = 1, numFeatures do
              dw[i] = 0
          end
          local db = 0
          for i = 1, numSamples do
              local prediction = sigmoid(sumOfProducts(weights, X[i]) + bias)
              local error = y[i] - prediction
              for j = 1, numFeatures do
                  dw[j] = dw[j] + error * X[i][j]
              end
              db = db + error
          end
          for j = 1, numFeatures do
              weights[j] = weights[j] + (learningRate * dw[j]) / numSamples
          end
          bias = bias + (learningRate * db) / numSamples
      end
      return weights, bias
  end

  local function sumOfProducts(arr1, arr2)
      if # arr1 ~= # arr2 then
          error("Arrays must have the same length")
      end
      local sum = 0
      for i = 1, # arr1 do
          sum = sum + arr1[i] * arr2[i]
      end
      return sum
  end

  -- Function to Predict using Logistic Regression
  local function predictLogistic(weights, bias, features)
      return sigmoid(sumOfProducts(weights, features) + bias)
  end
  local function CalculateRegularizedLinearRegressionnew(X, Y, regularizationType, lambda, learningRate, numIterations)
    -- Input: 
    --   X: A table of arrays, where each array represents a feature 
    --   Y: An array containing the target values 
    --   regularizationType: "lasso" or "ridge" 
    --   lambda: Regularization parameter
    --   learningRate: Learning rate for gradient descent
    --   numIterations: Number of iterations for gradient descent 
    -- Output:table of gains (slopes) for each feature and the offset (bias)
  
    -- Input Validation
    if not (X and Y and #X > 0 and #Y == #X) then
      error("Invalid input: X and Y should be tables of equal length.")
    end
  
    local numFeatures = #X[1]
    local numDataPoints = #Y
    lambda = lambda or 0.1
    learningRate = learningRate or 0.01
    numIterations = numIterations or 1000
  
    -- Initialize weights and bias
    local weights = {}
    for i = 1, numFeatures do
      weights[i] = 0
    end
    local bias = 0
  
    -- Helper function to calculate the dot product of two arrays
    local function dotProduct(a, b)
      if #a ~= #b then error("Arrays must be of same length for dot product") end
      local sum = 0
      for i = 1, #a do sum = sum + a[i] * b[i] end
      return sum
    end
  
    -- Gradient Descent with Regularization
    for iteration = 1, numIterations do
      local dw = {} 
      for i = 1, numFeatures do dw[i] = 0 end
      local db = 0
  
      for i = 1, numDataPoints do
        local prediction = dotProduct(weights, X[i]) + bias
        local error = prediction - Y[i]
  
        -- Calculate gradients
        for j = 1, numFeatures do
          dw[j] = dw[j] + error * X[i][j]
        end
        db = db + error
      end
  
      -- Update weights and bias
      for j = 1, numFeatures do
        if regularizationType == "lasso" then
          weights[j] = weights[j] - learningRate * ( (dw[j] / numDataPoints) + lambda * math.sign(weights[j]) )
        elseif regularizationType == "ridge" then
          weights[j] = weights[j] - learningRate * ( (dw[j] / numDataPoints) + 2 * lambda * weights[j] )
        else
          error("Invalid regularization type. Choose 'lasso' or 'ridge'.")
        end
      end
      bias = bias - learningRate * (db / numDataPoints)
    end
    return weights, bias
    end



-- Example Usage:

AOlearn = {
    linear_regression = {
        fit_linear = CalculateLinearRegression,
        predict_linear = CalculateResult
    },
    logistic = {
        fit_logistic = logisticRegression,
        predict_logistic = predictLogistic
      },
      lasso = {
        fit_lasso = function(X, Y, lambda, learningRate, numIterations)
              return CalculateRegularizedLinearRegressionnew(X, Y, "lasso", lambda, learningRate, numIterations)
          end,
          predict_lasso = CalculateResult -- Same prediction function as linear
      },
      ridge = {
          fit_ridge = function(X, Y, lambda, learningRate, numIterations)
              return CalculateRegularizedLinearRegressionnew(X, Y, "ridge", lambda, learningRate, numIterations)
          end,
          predict_ridge = CalculateResult -- Same prediction function as linear
      }
  }

  return AOlearn