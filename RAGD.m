function [GAMMA, label_sim] = RAGD(train_data, train_p_target)

param.lambda = 0.01;
param.mode = 2;
param.pos = true;

[rt, ct] = size(train_data);

step = rt/100;
count = 0;
steps = 100/rt;

GAMMA = zeros(rt, rt);

ins_cos = 1 - pdist2(train_data, train_data, 'cosine');  % 越相似，值越大
label_sim = 1 - pdist2(train_p_target, train_p_target, 'cosine'); % 判断候选标签集中是否有相同标签
label_sim(label_sim > 0) = 1;
label_sim = label_sim - diag(diag(label_sim));
sl = exp(1 - ins_cos .* label_sim); %origin

fprintf('generate RAGD\n');
for i = 1:rt
    if rem(i,step) < 1
		fprintf(repmat('\b',1,count-1));
		count = fprintf(1,'>%d%%',round((i+1)*steps));
    end
    W = sl(i,:)';
    W(i) = max(W);
    t = train_data(i,:)';
    tmp_data = train_data';
    tmp_data(:,i) = zeros(ct,1)';
    GAMMA(:,i) = mexLassoWeighted(t, tmp_data, W, param);
end
fprintf('\n');
GAMMA = GAMMA';
end
