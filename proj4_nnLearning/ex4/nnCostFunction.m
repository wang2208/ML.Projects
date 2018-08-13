function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);		% number of samples (rows) in X 
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Note: Theta1 dimension: 25 x 401; Theta2 dimension: 10 x 26;
%       X dimension: 5000 x 400; m = 5000 (# of samples) 

% forward propagation 
a1 = X;

X = [ones(m, 1) X];			% Add bias term to X, dimension now: 5000 x 401 
a2 = sigmoid(Theta1 * X');	% dimension: 25 x 5000

a2 = [ones(m, 1) a2']; 		% Add bias term to a2, dimension now: 5000 x 26
a3 = sigmoid(Theta2 * a2');	% dimension: 10 x 5000

H_theta = a3;

% recode the output labels as vectors, from dimension 5000 x 1, to 5000 x 10
y_matrix = eye(num_labels)(y, :);	% dimension 5000 x 10

% calc cost without regularized term 
J = -(1/m) * sum(sum((y_matrix' .* log(H_theta) + (1 - y_matrix') .* (log(1 - H_theta)))));

% add regularized term 
num_cols_1 = size(Theta1, 2);
num_cols_2 = size(Theta2, 2);
reg_theta1 = Theta1(:, 2 : num_cols_1);
reg_theta2 = Theta2(:, 2 : num_cols_2);

reg_term = lambda/(2*m)*(sum(sum(reg_theta1.^2)) + sum(sum(reg_theta2.^2)));

% calc cost with regularized term 
J = J + reg_term;


% backpropagation 
for t = 1:m
	
	% forward propagation 
	a1 = X(t, :)';			% 401 x 1, contains bias 
	z2 = Theta1 * a1;		% 25 x 1
	a2 = sigmoid(z2);
	a2 = [1; a2];			% add bias term, 26 x 1
	z3 = Theta2 * a2; 		% 10 x 1
	a3 = sigmoid(z3);
	
	% back propagation 
	delta3 = (a3 - y_matrix'(:, t));					% 10 x 1
	z2 = [1; z2]; 										% add bias term 
	delta2 = Theta2' * delta3 .* sigmoidGradient(z2);	% 26 x 1
	
	delta2 = delta2(2:end); 					%skip delta2(0), 25 x 1 
	
	Theta2_grad = Theta2_grad + delta3 * a2';	% 10 x 26
	Theta1_grad = Theta1_grad + delta2 * a1'; 	% 25 x 401
	
endfor

% Get the unregularized gradient for the neural network cost function 
Theta1_grad = 1/m .* Theta1_grad;
Theta2_grad = 1/m .* Theta2_grad;

% Add regularization (exclude bias terms)
Theta1_grad(:, 2:end) += lambda/m .* Theta1(:, 2:end);
Theta2_grad(:, 2:end) += lambda/m .* Theta2(:, 2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
