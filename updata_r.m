function r = updata_r(train_outputs, r, num_class, alpha)

% clear
% load('test data\updata_r_lost.mat');
% train_outputs = updata_r_lost.train_outputs;
% % r = updata_r_lost.r;
% num_class = updata_r_lost.num_class;
% num_data = size(train_outputs, 1);
% alpha = updata_r_lost.alpha;

% r = ones(num_class, 1)*num_data/num_class*2;

% main
[~,class] = max(train_outputs, [], 2);
tmp = tabulate(class);
z = zeros(num_class, 1);
num_outputs_class = size(tmp,1);
for i = 1:num_outputs_class
    z(tmp(i,1)) = tmp(i,2);
end

r = alpha*r + (1-alpha)*z;