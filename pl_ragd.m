function [accuracy, Precision, Recall, F_measure, MACU] = pl_ragd(train_data, train_p_target, test_data, test_target, optmParameter)

% main
% parameter
ker = optmParameter.ker; % Gaussian kernel
delta = optmParameter.delta;
Maxiter = optmParameter.Maxiter;
gama = optmParameter.gama;
beta = optmParameter.beta; % original
mu = optmParameter.mu;
k = optmParameter.k
[num_train, num_class] = size(train_p_target);

% initailze PL_target matrix 
y = build_label_manifold(train_data, train_p_target, k);
par = mean(pdist(train_data));

% RAGD
[GAMMA, ~] = RAGD(train_data, train_p_target);

%inital r
r = ones(num_class,1)*num_train/num_class*delta; % others

% inital Q
[train_outputs, test_outputs] = MulRegression(train_data, y, test_data, beta, par, ker);

for i = 1:Maxiter
    fprintf('The %d-th iteration\n',i);
    y = UpdateY(GAMMA, train_p_target,train_outputs,r,mu);
    [train_outputs, test_outputs] = MulRegression(train_data, y, test_data, beta, par, ker);
    r = updata_r(train_outputs, r, num_class, gama); 
end

accuracy = CalAccuracy(test_outputs, test_target);
[Precision, Recall, F_measure, MACU] = imbalance_loss(test_outputs, test_target, size(test_target,1), size(test_target,2));

% FG-NET(MAE-n)
% n = 3;
% num_test = size(test_data, 1);
% accuracy = Evaluation_MAE(test_target, test_outputs, num_test, n);
% fprintf('acc = %d\n', accuracy);

end
