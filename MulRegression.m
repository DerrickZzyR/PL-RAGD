function [train_outputs, test_outputs] = MulRegression(train_data, train_p_target, test_data, lambda, par, ker)
% this model is actually a multi-output kernel ridge regression model

% clear
% load('test data\malagasy_data.mat');
% train_data = malagasy_data.train_data;
% train_p_target = malagasy_data.train_p_target;
% test_data = malagasy_data.test_data;
% lambda = 0.05;
% par = mean(pdist(train_data));
% ker = 'rbf';

% main
[m, ~] = size(train_data);
[t, ~] = size(test_data);

K = kernelmatrix(ker,train_data',train_data',par); % m by m, kernel matrix
Kt = kernelmatrix(ker,test_data',train_data',par); 

I = eye(m, m);
H = (1/(2*lambda))*K+1/2*I; % m by m 
m1 = ones(m, 1);
s = (H\m1)'; % m by 1
P = train_p_target;
b = s*P/(s*m1);
alpha = H\(P-repmat(b, m, 1));

train_outputs = 1/(2*lambda)*K*alpha+repmat(b, m, 1); % formula(9) predicted label confidence(train)
test_outputs = 1/(2*lambda)*Kt*alpha+repmat(b, t, 1); % test result

end