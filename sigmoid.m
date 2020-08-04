function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

% sigmoid function using 'e' (euller's constant) raised to the power '-z'
% use either e.^ (element wise) or exp() - since z can be scalar or matrix
% g(z) = 1/(1+e.^-z) 
g = 1./(1 + exp(-z));

% =============================================================

end
