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


% size(X) = 100 3; size(y) = 100 1; size(theta) = numFeatures x 1

pred = sigmoid(X * theta);

part1 = y' * log (pred);
part2 = (1 - y') * log(1 - pred);

% remove theta0 effect 
theta0 = theta;
theta0(1) = 0;

reg_part = lambda / (2 * m) * sum (theta0 .^2);

J = - 1/m * (part1 + part2) + reg_part;	%number

grad = 1/m * X' * (pred - y)  + lambda / m * theta0;



% =============================================================

end
