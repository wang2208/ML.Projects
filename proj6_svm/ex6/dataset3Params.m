function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_init    =[0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigma_init=[0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

min_err = 1;   % init min error to a large value

for i = 1 : size(C_init, 1)
    for j = 1 : size(sigma_init, 1)
        model = svmTrain(X, y, C_init(i), ...
                         @(x1, x2) gaussianKernel(x1, x2, sigma_init(j)));
        predictions = svmPredict(model, Xval);
        cur_pred_err = mean(double(predictions ~= yval));
        if(cur_pred_err < min_err)  % get new min_err 
            min_err = cur_pred_err;
            C = C_init(i);
            sigma = sigma_init(j);
        endif
        %fprintf('i=%d, j=%d, cur_err=%f, C=%f, sigma=%f\n', ...
        %         i, j, cur_pred_err, C, sigma);
    end
end

%fprintf('Optimized C=%f, sigma=%f\n', C, sigma);

% =========================================================================

end
