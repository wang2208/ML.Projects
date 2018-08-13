function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Note: Theta1 size: 25 x 401 
%       Theta2 size: 10 x 26

% Add ones to the X data matrix
X = [ones(m, 1) X];

pred1 = sigmoid(Theta1 * X');	 % output from layer 1, size 25 x 5000

n = columns(pred1);
pred1 = [ones(1, n); pred1];	 % add bias and change size to 26 * 5000

pred2 = sigmoid(Theta2 * pred1); % output from layer 2, size 10 * 5000 

[max_val, max_idx] = max(pred2);
p = max_idx';


% =========================================================================


end
