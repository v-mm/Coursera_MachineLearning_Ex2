function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% in logistic regression
% cost = 1/m * SigmaSum(i = 1 to m)[-yilog(htheta(xi))-(1-yi)log(1-htheta(xi))]
% htheta(X) = g(theta'*X) = sigmoid (theta'*X),
% where X is (m x n+1), theta is (n+1 x 1) and y is (m x 1), htheta is (m x 1) 
% and cost is a scalar result
% SigmaSum of all i, is handled by vector multiplication inherently


% vector multiplication keeping in mind dimensions
% X * theta is (m x n+1) * (n+1 x 1) = (m x 1) 
HThetaXi = sigmoid(X * theta); % (m x 1) vector

_YiLogHThetaXi =  -y' * log(HThetaXi); 
% keeping dimensions in mind, result is scalar sum

One_YiLog1_HThetaXi = (ones(size(y)) - y)' * log(ones (size(HThetaXi))- HThetaXi);
% result is (1 x m) * (m x 1) = scalar 1 x 1

% cost
J = (1/m) * (_YiLogHThetaXi - One_YiLog1_HThetaXi);

% gradient calculation
% gradient = 1/m * SigmaSum(i = 1 to m)[htheta(xi)-yi]*xi
% result = (mx1)*(m x n+1) =>take as (n+1 x m)*(mx1) = (n+1 x 1) vector... 
% same as theta
% again SigmaSum of all i, is handled by vector multiplication inherently
 
grad = (1/m) * (X' * (HThetaXi - y));

% =============================================================

end
