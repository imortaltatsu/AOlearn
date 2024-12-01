local function sigmoid(z)
  return 1 / (1 + math.exp(-z))
end

local function dotProduct(a, b)
  if #a ~= #b then error("Arrays must be of same length for dot product") end
  local sum = 0
  for i = 1, #a do sum = sum + a[i] * b[i] end
  return sum
end

local function linearRegressionGradientDescent(X, y, learningRate, numIterations)
  local n = #X
  local m = #X[1] + 1              -- Add 1 for the bias term
  local theta = {}
  for i = 1, m do theta[i] = 0 end -- Initialize weights and bias to 0

  for iteration = 1, numIterations do
      local predictions = {}
      for i = 1, n do
          local sample = { 1 } -- Add 1 for bias term
          for j = 1, #X[i] do sample[#sample + 1] = X[i][j] end
          predictions[i] = dotProduct(theta, sample)
      end

      local gradients = {}
      for j = 1, m do gradients[j] = 0 end

      for i = 1, n do
          local error = predictions[i] - y[i]
          local sample = { 1 } -- Add 1 for bias term
          for j = 1, #X[i] do sample[#sample + 1] = X[i][j] end
          for j = 1, m do
              gradients[j] = gradients[j] + error * sample[j]
          end
      end

      for j = 1, m do
          theta[j] = theta[j] - learningRate * (gradients[j] / n)
      end
  end
  return theta
end

function math.sign(x)
  if x < 0 then return -1 end
  if x > 0 then return 1 end
  return 0
end

local function logisticRegression(X, y, learningRate, numIterations)
  local n = #X
  local m = #X[1]
  local weights = {}
  for i = 1, m do weights[i] = 0 end  -- Initialize weights to 0
  local bias = 0   --Initialize bias to zero

  for iteration = 1, numIterations do
    local dw = {}
    for i = 1, m do dw[i] = 0 end
    local db = 0

    for i = 1, n do
      local z = dotProduct(weights, X[i]) + bias
      local prediction = sigmoid(z)
      local error = prediction - y[i] --Using prediction - y[i] for consistency
      for j = 1, m do
        dw[j] = dw[j] + error * X[i][j]
      end
      db = db + error
    end
    for j = 1, m do
        weights[j] = weights[j] - learningRate * dw[j] / n
    end
    bias = bias - learningRate * db / n

  end
    return weights, bias
end

local function CalculateRegularizedLinearRegression(X, y, regularizationType, lambda, learningRate, numIterations)
  local n = #X
  local m = #X[1] + 1                -- Add 1 for the bias term
  local theta = {}
  for i = 1, m do theta[i] = 0 end   -- Initialize weights and bias to zero

  for iteration = 1, numIterations do
      local predictions = {}
      for i = 1, n do
          local sample = { 1 }
          for j = 1, #X[i] do sample[#sample + 1] = X[i][j] end
          predictions[i] = dotProduct(theta, sample)
      end

      local gradients = {}
      for j = 1, m do gradients[j] = 0 end

      for i = 1, n do
          local error = predictions[i] - y[i]
          local sample = { 1 }
          for j = 1, #X[i] do sample[#sample + 1] = X[i][j] end
          for j = 1, m do
              gradients[j] = gradients[j] + error * sample[j]
          end
      end

      --Update weights with regularization term
      for j = 1, m do
          local regTerm = 0
          if regularizationType == "lasso" then
              regTerm = lambda * math.sign(theta[j])
          elseif regularizationType == "ridge" then
              regTerm = 2 * lambda * theta[j]
          end

          theta[j] = theta[j] - learningRate * (gradients[j] / n + regTerm)
      end
  end

  return theta
end

local AOlearn = {
  linear_regression = {
      fit_linear = function(X, y, learningRate, numIterations)
          local theta = linearRegressionGradientDescent(X, y, learningRate, numIterations)
          return theta   -- Return coefficients
      end,

      predict_linear = function(theta, features)
          local sample = { 1 } -- Add bias term
          for i = 1, #features do sample[#sample + 1] = features[i] end -- Add features
          return dotProduct(theta, sample)
      end
  },

  logistic = {
      fit_logistic = function(X, y, learningRate, numIterations)
          return logisticRegression(X, y, learningRate, numIterations)
      end,
      predict_logistic_sigmoid = function(weights, bias, features)
          return sigmoid(dotProduct(weights, features) + bias)
      end

  },

  lasso = {
      fit_lasso = function(X, y, lambda, learningRate, numIterations)
          return CalculateRegularizedLinearRegression(X, y, "lasso", lambda, learningRate, numIterations)
      end,
      predict_lasso = function(theta, features)
          local sample = { 1 } -- Add bias term
          for i = 1, #features do sample[#sample + 1] = features[i] end -- Add features
          return dotProduct(theta, sample)
      end
  },

  ridge = {
      fit_ridge = function(X, y, lambda, learningRate, numIterations)
          return CalculateRegularizedLinearRegression(X, y, "ridge", lambda, learningRate, numIterations)
      end,
      predict_ridge = function(theta, features)
          local sample = { 1 } -- Add bias term
          for i = 1, #features do sample[#sample + 1] = features[i] end -- Add features
          return dotProduct(theta, sample)
      end
  }
}

return AOlearn
