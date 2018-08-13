function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%


% Find Indices of Positive and Negative Examples
pos = find(y==1); neg = find(y == 0);
% Plot Examples
X1p = X(pos, 1);  % get first column positive
X2p = X(pos, 2);  % get second column positive
X1n = X(neg, 1);  % get first column negative
X2n = X(neg, 2);  % get second column negative
plot(X1p, X2p, 'k+', 'LineWidth', 2, 'MarkerSize', 7);
plot(X1n, X2n, 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);


% =========================================================================



hold off;

end
