function [ibi3, bi3] = mibi3(data, label)
%calculate ibi3、bi3 for PLL

num_label = size(label,1);

ibi3 = cell(num_label,1);
bi3 = zeros(num_label,1);

for j = 1:num_label
    fprintf('label = %d\n', j)
    pos_num = sum(label(j,:) == 1);
    neg_num = sum(label(j,:) == 0); %label == -1
    pos_idx = find(label(j,:) == 1);
    neg_idx = find(label(j,:) == 0); %label == -1

    pos_data = data(pos_idx, :);

    rr = neg_num / pos_num;

    k = 5;
    % find k neighbors
    kdtree = KDTreeSearcher(data);
    [knn_idx, ~] = knnsearch(kdtree,data,'k',k+1);

    p2 = zeros(1, pos_num);
    p2old = zeros(1, pos_num);
    knn_idx(:,1) = [];
    % calculate IBI3
    temp_num2 = zeros(1, pos_num);

    for i = 1:pos_num
        temp_num = size(intersect(knn_idx(i,:), neg_idx), 2);
        temp_num2(i) = temp_num;
        p2(i) = temp_num / k;
        p2old(i) = p2(i);
        if p2(i) == 1 % 近邻全为负样本
            fprintf('第%d个样本的近邻全为负样本\n',i)
            dd = pdist2(pos_data(i,:), data);
            [~, sort_idx] = sort(dd);
            nearest_pos = find(label(sort_idx) == 1); % 找出距离样本i最近的正样本下标
            p2(i) = (nearest_pos(1) -1) / nearest_pos(1);
        end
    end

    p1 = 1 - p2;
    ibi3{j} = (rr * p1 / (p2 + rr * p1) - p1);
    bi3(j) = mean(ibi3{j});

end

end
