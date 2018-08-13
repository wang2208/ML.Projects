function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h_theta = X * theta;
% Cost without regularization 
J = 1/(2*m)*(h_theta-y)'*(h_theta-y);

% Regularization term
reg_theta = theta(2 : end);	%remove theta0
reg = lambda/(2*m)*reg_theta'*reg_theta;

J += reg;	% Reg cost 

% Gradient
grad         = 1/m * X' * (h_theta - y);	 %j = 0
grad(2:end) += lambda/m * reg_theta;         %j >= 1


% =========================================================================

grad = grad(:);

end
