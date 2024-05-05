function Outputs = UpdateY(W, train_p_target,train_outputs,r,mu)
% Update label confidence Y

[p,q]=size(train_p_target);

options = optimoptions('quadprog',...
'Display', 'off','Algorithm','interior-point-convex' );

WT = W';
% WT = WT./sum(WT);
sum(WT);
% tic
%T =WT*W+ W*ones(p,p)*WT.*eye(p,p)-2*WT+ 1/mu*eye(p);
T = 2*(eye(p)-W)'*(eye(p)-W)+2/mu*eye(p);
% T = 2*((mu + 1));
%T(1:10,1:10)
% toc
T1 = repmat({T},1,q);
M = spblkdiag(T1{:});
%M = M +2/mu*eye(p*q);
lb = sparse(p*q,1);
ub = reshape(train_p_target,p*q,1);
II = sparse(eye(p));
A = zeros(q,p*q);
for i = 1:q
    A(i,i:q:p*q) = 1;
end
b = r;
Aeq = repmat(II,1,q);
beq=ones(p,1);
% M = (M+M');

f = reshape(train_outputs, p*q, 1);
Outputs = quadprog(M, -2*(1/mu)*f, A, b, Aeq, beq, lb, ub,[], options);
% Outputs2 = quadprog(M, -2*(1/mu)*f, [], [], Aeq, beq, lb, ub,[], options);
Outputs = reshape(Outputs,p,q);

end