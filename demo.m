clear
load('lost.mat');
data = lost.data;
target = lost.target;
partial_target = lost.partial_target;

preprocess = 2;
data = DataSegment(data, preprocess);
tr_idx = lost.tr_idx;
te_idx = lost.te_idx;
optmParameter = lost.optmParameter;

Miter = 10;
acc = zeros(Miter,1);
Precision = zeros(Miter,1);
Recall = zeros(Miter,1);
F_measure = zeros(Miter,1);
MACU = zeros(Miter,1);

for i = 1:10
    train_data = data(tr_idx(:,i),:);
    train_p_target = partial_target(:,tr_idx(:,i))';
    train_target = target(:,tr_idx(:,i))';
    test_data = data(te_idx(:,i),:);
    test_target = target(:,te_idx(:,i))';
    [acc(i), Precision(i), Recall(i), F_measure(i), MACU(i)] = pl_ragd(train_data, train_p_target, test_data, test_target, optmParameter);
end

macc = mean(acc);
mPrecision = mean(Precision);
mRecall = mean(Recall);
mF_measure = mean(F_measure);
mMACU = mean(MACU);