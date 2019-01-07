function [U_1,U_2] = metric_learning (I,T,labels)
% Learning the mapping U_1 and U_2 
% Input:
% I - the image feature, the column is the number of feature and the row is
%     the number of image;
% T - the txt feature, the column is the number of feature and the row is
%     the number of text;
% labels - the labes for each pair, the row is the image and the column
%            is the text

[para.I_ins_num,para.I_fe_num] = size(I);
[para.T_ins_num,para.T_fe_num] = size(T);

% para.d = min(para.I_fe_num,para.T_fe_num); % the feture number of the common space U, it can be any number
para.d = 10;
para.max_iter = 100;
para.eps = 1e-3;
para.labels = labels;

[U_1,U_2] = mtl_grid( para,I,T );
end

function [U_1,U_2] = mtl_grid( para,I,T )
% select the best U_1 and U_2 through the grid search
base = 10; %
n = 4; %
m = 4;

result_eval = cell(n+1,m+1);
for i = 1 : n+1
    for j = 1 : m+1
        result_eval{ i, j } = Inf;
    end;
end;
    
para1 = base.^(-5+(1:n)); % 
para2 = base.^(-5+(1:m));
for i = 1 : n
    result_eval{ i+1, 1 } = para1(i);
end
for j = 1 : m
   result_eval{ 1, j+1 } = para2(j);
end

% 
U = cell(m,n);
for j = 1 : m
    for i = 1:n
        para.para1 =  para1(i);
        para.para2 = para2(j);
        U{i,j} = mtl_func(para,I,T);   
        result_eval{ i+1, j+1 } = mtl_eval(U{i,j},I,T,para);            
    end;
end;

% 
[~,idx] = min([result_eval{2:end,2:end}]);
idx1 = rem(idx-1, size(result_eval,1)-1)+1;
idx2 = floor((idx-1)/ (size(result_eval,1)-1))+1;
U_1 = U{idx1,idx2}.U_1;
U_2 = U{idx1,idx2}.U_2;
end

function U = mtl_func(para,I,T)
% Given the parameters, learn the mapping U_1 and U_2
% intilize A, U_1, U_2
d = para.d;
I_ins_num = para.I_ins_num;
I_fe_num = para.I_fe_num;
T_ins_num = para.T_ins_num;
T_fe_num = para.T_fe_num;
eps = para.eps;
max_iter = para.max_iter;
labels = para.labels;

% intilize A, U_1, U_2
A = eye(d);
U_1 = rand(I_fe_num,d);
U_2 = rand(T_fe_num,d);
% normalization
U_1 = U_1 * diag(sqrt(1 ./ (sum(U_1.^2) + eps)));
U_2 = U_2 * diag(sqrt(1 ./ (sum(U_2.^2) + eps)));

X_I = I * U_1; % each row is one instance
X_T = T * U_2;

% compute D_0 D_1
tmp_D_0 = zeros(size(A));
tmp_D_1 = zeros(size(A));
for i = 1:I_ins_num
    for j = 1:T_ins_num
        if labels(i,j) == 0
            tmp_D_0 = tmp_D_0 + (X_I(i,:)-X_T(j,:))' * (X_I(i,:)-X_T(j,:));
        else
            tmp_D_1 = tmp_D_1 + (X_I(i,:)-X_T(j,:))' * (X_I(i,:)-X_T(j,:));
        end
    end
end

% update A, U_1, U_2
tmp_F = compute_F (para,A,U_1,U_2,I,T);
for l = 1:max_iter;  
    F = tmp_F;
    if rem(l,3)==1
        F_disp = ['iteration = ' num2str(l) ': ' 'F = ' num2str(F) ];
        disp(F_disp);
    end
    tmp_A = A;
    A = update_A (para,A,tmp_D_0,tmp_D_1);
    [~,tmp_A_idx] = chol(A);
    if tmp_A_idx ~= 0
        A = tmp_A;
    end
    U_1 = update_U_1 (para,A,U_1,X_I,X_T,I);
    U_2 = update_U_2 (para,A,U_2,X_I,X_T,T);
    tmp_F = compute_F (para,A,U_1,U_2,I,T);
    erro = F - tmp_F;
    if erro <= eps  
        break;
    end
end

U.U_1 = U_1;
U.U_2 = U_2;
U.A = A;

end

function result_eval = mtl_eval(U,I,T,para)
% compute the evaluation result
I_ins_num = para.I_ins_num;
T_ins_num = para.T_ins_num;
labels = para.labels;

X_I = I * U.U_1;
X_T = T * U.U_2;
A = U.A;

D = zeros(I_ins_num,T_ins_num);
for i = 1:I_ins_num
    for j = 1:T_ins_num
        D(i,j) = sqrt((X_I(i,:)-X_T(j,:)) * A * (X_I(i,:)-X_T(j,:))');
    end
end

D = 1./(1 + exp(D));
result_eval = sqrt(sum(sum((D-labels).^2)));
% 
% D_max = max(max(D));
% D_min = min(min(D));
% D_eps = D_min + (D_max - D_min) * sum(sum(labels))/(I_ins_num * T_ins_num);
% 
% tmp_D = D;
% D(tmp_D<D_eps) = 1;
% D(tmp_D>=D_eps) = 0;
% 
% result_eval = sqrt(sum(sum((D-labels).^2)));
end

function A = update_A (para,A,D_0,D_1)
% update matrix B
A_step = 0.1;
gamma_0 = para.para1;
gamma_1 = para.para2;

paratal_A = compute_paratal(A);
Delta_A = paratal_A - gamma_0 * D_0' + gamma_1 * D_1';

A = A - A_step * Delta_A;
end

function U_1 = update_U_1 (para,A,U_1,X_I,X_T,I)
% update matrix U_1
U_step = 0.1;
gamma_0 = para.para1;
gamma_1 = para.para2;
I_ins_num = para.I_ins_num;
T_ins_num = para.T_ins_num;
labels = para.labels;

paratal_U_1 = compute_paratal(U_1);
paratal_D_0 = zeros(size(U_1));
paratal_D_1 = zeros(size(U_1));
for i = 1:I_ins_num
    for j = 1:T_ins_num
        if labels(i,j) == 0
            paratal_D_0 = paratal_D_0 + 2 * I(i,:)' * (X_I(i,:)-X_T(j,:));
        else
            paratal_D_1 = paratal_D_1 + 2 * I(i,:)' * (X_I(i,:)-X_T(j,:));
        end
    end
end


Delta_U_1 = paratal_U_1 - gamma_0 * paratal_D_0  * A + gamma_1 * paratal_D_1 * A;

U_1 = U_1 - U_step * Delta_U_1;
end

function U_2 = update_U_2 (para,A,U_2,X_I,X_T,T)
% update matrix U_2
U_step = 0.1;
gamma_0 = para.para1;
gamma_1 = para.para2;
I_ins_num = para.I_ins_num;
T_ins_num = para.T_ins_num;
labels = para.labels;

paratal_U_2 = compute_paratal(U_2);
paratal_D_0 = zeros(size(U_2));
paratal_D_1 = zeros(size(U_2));
for i = 1:I_ins_num
    for j = 1:T_ins_num
        if labels(i,j) == 0
            paratal_D_0 = paratal_D_0 - 2 * T(j,:)' * (X_I(i,:)-X_T(j,:));
        else
            paratal_D_1 = paratal_D_1 - 2 * T(j,:)' * (X_I(i,:)-X_T(j,:));
        end
    end
end

Delta_U_2 = paratal_U_2 - gamma_0 * paratal_D_0 * A + gamma_1 * paratal_D_1 * A;

U_2 = U_2 - U_step * Delta_U_2;
end

function F = compute_F (para,A,U_1,U_2,I,T)
% compute the objective F
gamma_0 = para.para1;
gamma_1 = para.para2;
labels = para.labels;
I_ins_num = para.I_ins_num;
T_ins_num = para.T_ins_num;

X_I = I * U_1;
X_T = T * U_2;

tmp_D_0 = zeros(size(A));
tmp_D_1 = zeros(size(A));
for i = 1:I_ins_num
    for j = 1:T_ins_num
        if labels(i,j) == 1
            tmp_D_0 = tmp_D_0 + (X_I(i,:)-X_T(j,:))' * (X_I(i,:)-X_T(j,:));
        else
            tmp_D_1 = tmp_D_1 + (X_I(i,:)-X_T(j,:))' * (X_I(i,:)-X_T(j,:));
        end
    end
end

F = trace(A' * A) + trace(U_1' * U_1) + trace(U_2' * U_2)...
   + gamma_1 * trace(A * tmp_D_1);
end

function paratal_B = compute_paratal(B)
% compute the paratal of matrix B
b = sqrt(sum(B.^2,1));
paratal_B = B./(repmat(b,size(B,1),1) + 1e-6);
end