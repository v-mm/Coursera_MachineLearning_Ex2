function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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


% in regularized logistic regression
% cost = (1/m)*SigmaSum(i=1 to m)[-yilog(htheta(xi))-(1-yi)log(1-htheta(xi))]...
% ...+ Regularization Term i.e. RegTermCost 
% and RegTermCost = (1/2m)*lambda*SigmaSum(i=1 to m)[theta_squared]
% and theta_squared = theta*theta keeping in mind dimensions
% and the vector multiplication takes care of the SigmaSum

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

% set Theta_0 to 1; since Theta_0 is not regularized
% note:theta here is a local variable; at this point HThetaXi is already calculated
theta(1) = 0;
% theta is (n+1 x 1), theta' is (1 x n+1)
RegTermCost = (1/m) * (1/2) * lambda * (theta' * theta);

% cost
J = ((1/m) * (_YiLogHThetaXi - One_YiLog1_HThetaXi)) + RegTermCost;

% regularized gradient calculation
% gradient = 1/m * SigmaSum(i = 1 to m)[htheta(xi)-yi]*xi + Regularization Term
% where RegTerm = 1/m * lambda * theta_j for J>0,for j=0, RegTerm = 0
% [htheta(xi)-yi]*xi = (mx1)*(m x n+1) =>take as (n+1 x m)*(mx1) = (n+1 x 1)...  
% same as theta
% RegTerm = scalar value * theta
% result = (n+1 x 1) + (n+1 x 1) = (n+1 x 1) 
% again SigmaSum of all i, is handled by vector multiplication inherently

% note - Theta_0 is not regularized, so here too theta(1) = 0
RegTermGrad = (1/m) * lambda * theta;

grad = (1/m) * (X' * (HThetaXi - y)) + RegTermGrad;


% =============================================================

end
