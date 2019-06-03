function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

z = X * theta;
h = sigmoid(z);
% for i = 1 : size(theta,1)
%     if i == 1
%         grad(i) = sum((h - y) .* X(:,i)) ./ m;
%     else
%         grad(i) = sum((h - y) .* X(:,i)) ./ m + lambda / m * theta(i);
%     
%     end
% 
% end
J = -(y' * log(h) + (1 - y)' * log(1-h)) / m + (theta(2:end)' * theta(2:end))  * lambda / 2 / m;
grad = X' * (h - y) / m + lambda / m .* theta;
grad(1) = X(:,1)' * (h - y) ./ m;
% =============================================================
end
