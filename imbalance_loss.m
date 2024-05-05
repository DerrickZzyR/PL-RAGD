function [Precision, Recall, F, MACU] = imbalance_loss(predict_label, test_target, num_data, num_class)
% precision = TP / (TP+FP)
% recall = TP / (TN+FP)
% specificity = FN / (TN+FP)
% F = (2*recall*precision) / (recall+precision)

pseudo_test_target = zeros(num_data, num_class);

[~, te_ind1] = max(predict_label, [], 2);
% [~, te_ind2] = max(test_target, [], 2);

for i = 1:num_data
    pseudo_test_target(i, te_ind1(i)) = 1;
end

p = zeros(1, num_class);
r = zeros(1, num_class);
f = zeros(1, num_class);

for i = 1:num_class
    TP_te = pseudo_test_target(:,i) .* test_target(:,i);
    FP_te = pseudo_test_target(:,i) .* ~test_target(:,i);
    FN_te = test_target(:,i) .* ~pseudo_test_target(:,i);
    TN_te = ~test_target(:,i) .* ~pseudo_test_target(:,i);
    TP = sum(TP_te);
    FP = sum(FP_te);
    FN = sum(FN_te);
    TN = sum(TN_te);

    if TP == 0
        p(i) = 0;
        r(i) = 0;
    else
        p(i) = TP / (TP + FP);
        r(i) = TP / (TP + FN);
    end

    if p(i) == 0 & r(i) == 0
        f(i) = 0;
    else
        f(i) = 2*p(i)*r(i) / (p(i)+r(i));
    end
end
Precision = mean(p);
Recall = mean(r);
F = mean(f);
MACU = calMAUC(test_target', pseudo_test_target', predict_label);
