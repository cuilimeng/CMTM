classdef slexpclassifierMFM < slexpclassifier
    properties
        lambda_; % regularization parameter
        gamma1_; % regularization parameter
        regular_task;
        regular_view;
        regular_u;
        k_; % number of factors
        iteration_;        
        para_train='-lambda: 1e-3 -gamma: 1e-3 -k: 20';
    end
    
    methods
        function s = slexpclassifierMFM()
            s = s@slexpclassifier( 'MFM', 'multilinear factorization machines' );
            s.discription = 'MFM';
        end
        function [ Outputs, Pre_Labels, s ] = classify( s, train_data, train_label, test_data, view_index )
            % data is a T * 1 cell matrix; each cell is a [d_1;...;d_V ] * n_t matrix
            % view_index is a vector of the indices of each view in the data
            % labels is a T * 1 cell matrix
            rand ( 'seed', 5948 );
            k = s.k_;
            eps = 1e-6;
            regPara.lambda = s.lambda_;
            regPara.gamma_1 = s.gamma1_;
            if any(strcmp(fieldnames(s), 'regular_task') )
                Para.regular_task = s.regular_task;
            else
                Para.regular_task = 'L2';
            end
            if any(strcmp(fieldnames(s), 'regular_view') )
                Para.regular_view = s.regular_view;
            else
                Para.regular_view = 'L2';
            end
            if any(strcmp(fieldnames(s), 'regular_u') )
                Para.regular_u = s.regular_u;
            else
                Para.regular_u = 'L2';
            end
            
            num_iter  = s.iteration_;
            num_task = length(train_label);
            num_view  = length( view_index )/2;
            view_index = [ 0, view_index];
            
            
            Xtrain = [];
            Xtest = [];
            Ytrain = [];

            train_index = zeros(num_task+1,1);
            test_index = zeros(num_task+1,1);
            %
            for t = 1:num_task
                train_index(t+1) = train_index(t)+size( train_data{t}, 2 );
                test_index(t+1) = test_index(t)+size( test_data{t} , 2 );

                Xtrain = [Xtrain train_data{t}];
                Xtest = [Xtest test_data{t}];
                
                Ytrain = [Ytrain; full(train_label{t})']; 
            end
            clear train_data test_data train_label;
            
            running_t=cputime;            
            % initialize 
            Para.alpha = 0.5;
            Para.Theta = cell( num_view, 1 );
            Ada.Theta = cell( num_view, 1 );
            Para.Phi = randn( num_task, k );
            Ada.Phi = zeros( size(Para.Phi));
            
            num_fea = size(Xtrain,1);
            Para.U = randn(num_task,num_fea);
            Ada.U = zeros( size(Para.U));
            % assign zeros to missing features in each task
            for t = 1:num_task
                idx = train_index(t)+1:train_index(t+1);
                missing_features = sum(Xtrain(:,idx)>0,2)==0;
                Para.U(t,missing_features) = 0;
            end
            %Initialize Theta{v}
            for v = 1 : num_view
                tmp = randn( view_index( v + 1 ) - view_index(v) + 1....
                    + view_index( v + 4 ) - view_index(v+3) +1, k ); %
                Para.Theta{v} = tmp * diag(sqrt(1 ./ (sum(tmp.^2) + eps)));%
                Ada.Theta{v} = zeros(size(Para.Theta{v}));
            end;
            
            history = zeros(num_iter,1);
            regPara.step_size = 0.1;
            for  iter = 1: num_iter
                for v = 1 : num_view
                    [Para, Ada] = Update_Theta(Para, Ada, regPara, Xtrain, Ytrain, view_index,train_index, v);
                end;                    
                [Para, Ada] = Update_Phi(Para, Ada, regPara, Xtrain, Ytrain,view_index,train_index);
                [Para, Ada] = Update_U(Para, Ada, regPara, Xtrain, Ytrain, view_index,train_index);                   

                history(iter) = Compute( Para, regPara, Xtrain, Ytrain, view_index,train_index);
                if rem(iter,20)==1
                    fprintf( '(%d):\t%.6f\n', iter, history( iter ) );
                end
                if isnan( history( iter ) ) || ( iter > 1 && history( iter ) > history( iter - 1 ) - 1e-6 )
                    break;
                end;
            end
            
            s.time_train = cputime-running_t;
            s.time = cputime - running_t;            
%             plot( history );
            
            % test
            running_t=cputime;
            
            [Outputs_task, ~, ~] = Predict( Para, Xtest, view_index, test_index, -1 );
            Outputs_task = Outputs_task';
            
            Pre_Labels_task = Outputs_task;
%             Outputs_task = 1 ./ ( 1 + exp( - Outputs_task) ); 
%             Pre_Labels_task = -1*ones(size(Outputs_task));
%             Pre_Labels_task( Outputs_task > 0.5 ) = 1;
            
            
            Pre_Labels = cell(num_task,1);
            Outputs = cell(num_task,1);            
            for t= 1:num_task
                idx = test_index(t)+1:test_index(t+1);
                Pre_Labels{t} = Pre_Labels_task(idx);
                Outputs{t} = Outputs_task(idx);
            end    
            

            s.time_test = cputime-running_t;

            s.time = s.time_train + s.time_test;
            s.para_train = ['-lambda:' num2str(regPara.lambda) ' -gamma:' num2str(regPara.gamma_1) ' -k:' num2str(s.k_)];
            % save running state discription
            s.abstract=[s.name  '('...
                        '-time:' num2str(s.time)...
                        '-time_train:' num2str(s.time_train)...
                        '-time_test:' num2str(s.time_test)...
                        '-para:' s.para_train ...
                        ')'];
                    
        end
    end
end

function [ Para, Ada ] = Update_Phi( Para, Ada, regPara, X, Y, view_index, task_index)
% grad_phi = sum_i E(:,i)^T * delta_L(i) * Pi_Z_Theta(i,:);
% Phi <-  Phi - step_size * grad_phi ./ sqrt(Ada.Phi);
% Pi_Z_Theta: is a T*1 cell array of n_t * k matrix, where each row is product of Z_Theta
    lambda = regPara.lambda;
    
    step_size = regPara.step_size;
    [ S, ~, Pi_Z_Theta ] = Predict( Para, X, view_index, task_index, -1);
    [~, delta_L] = Compute_loss(S, Y, task_index);
    Phi = Para.Phi;

    delta_loss = zeros(size(Phi));   
    num_task = length(task_index)-1;
    for t = 1:num_task
        idx = task_index(t)+1:task_index(t+1);
        delta_loss(t,:) = delta_L(idx)' * Pi_Z_Theta(idx,:);        
    end
    
    if strcmp(Para.regular_task,'L1')
        grad = delta_loss + lambda * (Phi ./ (Phi.^2+eps));
    elseif strcmp(Para.regular_task,'L21')
        D_Phi = ComputeDv(Phi);
        grad = delta_loss + lambda * D_Phi;
    else % default is L2
        grad = delta_loss + 2*lambda* Phi;
    end
    Ada.Phi = Ada.Phi + power( grad, 2 );
    Para.Phi = Para.Phi - step_size * grad ./ ( sqrt( Ada.Phi ) + 1e-6 );
end

function [ Para, Ada ] = Update_Theta( Para, Ada, regPara, X, Y, view_index, task_index, view)
% delta_loss = sum_i z_view(:,i) * delta_L(i) * Pi_Z_Theta_v(i,:) * Phi(t,:);
% grad_theta = delta_loss + lambda * Theta
% Theta <-  Theta - step_size * grad_theta ./ sqrt(Ada.Theta);
% Pi_Z_Theta_v: is a N * k matrix, 
%    where each row is the product of Z^{~which_view} * theta^{~which_view}
    step_size = regPara.step_size;
    lambda = regPara.lambda;
    num_view = (length(view_index)-1)/2;
    
    [ S, Pi_Z_Theta_v, ~] = Predict( Para, X,  view_index, task_index, view);
    [~, delta_L] = Compute_loss(S, Y, task_index);
    Theta = Para.Theta{view};%
    
    I_idxs = view_index(view)+1:view_index(view+1);    
    T_idxs = view_index(view + num_view)+1:view_index(view+ num_view + 1);
    N = size(X,2);
    Z_I_view = [Para.alpha * ones(1,N); X(I_idxs,:)];%��view��Z_I(v)
    Z_T_view = [Para.alpha * ones(1,N); X(T_idxs,:)];%��view��Z_T(v)
    Z_view = [Z_I_view;Z_T_view];
    delta_loss = zeros(size(Theta));    

    num_task = length(task_index)-1;
    for t = 1:num_task
        idx = task_index(t)+1:task_index(t+1);
        tmp = bsxfun(@times, Pi_Z_Theta_v(idx,:),Para.Phi(t,:));
        tmp = bsxfun(@times, tmp, delta_L(idx,:));
        delta_loss = delta_loss + Z_view(:,idx) * tmp;
    end
    
    if strcmp(Para.regular_view,'L1')
        grad_theta = delta_loss + lambda * (Theta ./ (Theta.^2+eps));
    elseif strcmp(Para.regular_view,'L21')
        D_Theta = ComputeDv(Theta);
        grad_theta = delta_loss + lambda * D_Theta;
    else % default is L2
        grad_theta = delta_loss + 2*lambda* Theta;
    end
   
    
    Ada.Theta{view} = Ada.Theta{view} + power( grad_theta, 2 );
    Para.Theta{view} = Para.Theta{view} - step_size * grad_theta ./ ( sqrt( Ada.Theta{view} ) + 1e-6 );
%     Para.Theta{view} = Para.Theta{view} - step_size * grad_theta;
end

function [ Para, Ada ] = Update_U( Para, Ada, regPara, X, Y, view_index, task_index)
% U is a D * T matrix 
% f(x) = z^T*U + <W,Z>
% delta_loss_t = sum_t,i X(:,i) * delta_L(i);
% grad_U_t = delta_loss + gamma * D*U_t
% grad_U_t <-  grad_U_t - step_size * grad_theta ./ sqrt(Ada.Theta);
% Pi_Z_Theta: is a N * k matrix, where each row is product of Z_Theta
    gamma_1 = regPara.gamma_1;
    eps = 1e-6;
    step_size = regPara.step_size;
    [ S, ~, ~] = Predict( Para, X, view_index, task_index, -1);
    [~, delta_L] = Compute_loss(S, Y, task_index);
    
    U = Para.U;
    % 2,1 norm is column sparse
%    D_dash = ComputeDv(U,eps);
    
    % Z = D*N features
    num_task = length(task_index)-1;
    delta_loss = zeros(size(U));
    
    for t = 1:num_task
        idx = task_index(t)+1:task_index(t+1);
        delta_loss(t,:) = delta_L(idx)' * X(:,idx)';
    end
    
    if strcmp(Para.regular_u,'L1')
        grad_U = delta_loss + gamma_1 * (U ./ (U.^2+eps));
    elseif strcmp(Para.regular_u,'L21')
        D_U = ComputeDv(U);
        grad_U = delta_loss + gamma_1 * D_U;
    else % default is L2
        grad_U = delta_loss + 2*gamma_1* U;
    end
    
%     grad_U = delta_loss + gamma_1 * D_dash* U;
    
    Ada.U = Ada.U + power( grad_U, 2 );
    Para.U = Para.U - step_size * grad_U ./ ( sqrt( Ada.U ) + 1e-6 );
%     Para.U = Para.U - step_size * grad_U;
end

function [F] = Compute( Para, regPara, X, Y, view_index, task_index)
% F: value of the objective function
% F = \sum_i loss_i
%   + \lambda ( |Phi|_F^2 + \sum_p^V |Theta^{p}|_F^2)
%   + \gamma * |U|_21;
    lambda = regPara.lambda;
    gamma_1 = regPara.gamma_1;
        
    num_view = length(Para.Theta);
    [ S, ~, ~] = Predict( Para, X, view_index, task_index, -1 );
    [F, ~] = Compute_loss(S, Y, task_index);
%     k = size( Para.Phi, 2 );
%     I = eye(k);
    
    F = F + lambda * ComputeNorm(Para.Phi,Para.regular_task);
    for v = 1 : num_view
        F = F + lambda * ComputeNorm(Para.Theta{v},Para.regular_view);
    end;    
    F = F +  gamma_1 * ComputeNorm(Para.U,Para.regular_u);
end

function [F, delta_L] = Compute_loss(S,Y, task_index)
% require predicted values S, and ground truth Y
% F: summation of loss
% delta_L: a n_t*1 array of the gradient of normalized loss
    logit2 = @(x) 1./(1+exp(-x));
    
    delta_L = 2 * ( S - Y );
    loss = power( Y - S, 2 ); 
%     delta_L = - Y./(logit2(S)) + (1-Y)./(1-logit2(S));
%     loss = -(Y.*log(logit2(S)) + (1-Y).*log(1-logit2(S)));

    
    num_task = length(task_index)-1;
    F = 0;
    for t = 1:num_task
        N_t = task_index(t+1)-task_index(t);
        idx = task_index(t)+1:task_index(t+1);
        F = F + sum(loss(idx))/N_t;
        delta_L(idx) = delta_L(idx) / N_t;
    end
end

function [ S, Pi_Z_Theta_v, Pi_Z_Theta ] = Predict( Para, X, view_index, task_index, which_view )
% Z: is a  D*N matrix
% S: is a 1*N vector of predictied values 
% Pi_Z_Theta_v: is a N * k matrix, 
%       where each row is the product of Z^{~which_view} * theta^{~which_view}
% Pi_Z_Theta: is a N * k matrix, where each row is product of Z_Theta 
    num_view = length(Para.Theta);
    k = size( Para.Phi,2);
    N = size(X,2);
    
    Pi_Z_Theta_v = ones(N,k);
    Pi_Z_Theta = ones(N,k);
    
    for v = 1 : num_view
        if v+1 > length(view_index)
            disp(v);
        end
        if size(X,1) < view_index(v+1)
            disp(view_index); 
        end
        tmp_I = [Para.alpha * ones(1,N); X(view_index(v)+1:view_index(v+1),:)];%Z_I(v)
        tmp_T = [Para.alpha * ones(1,N); X(view_index(v+num_view)+1:view_index(v+num_view+1),:)];%Z_T(v)
        tmp = [tmp_I;tmp_T];%Z(v)
        tmp  = tmp' * Para.Theta{v};%<Z(v),Theta(v)>
        if (v ~= which_view)
            Pi_Z_Theta_v = Pi_Z_Theta_v.*tmp;
        end
        Pi_Z_Theta = Pi_Z_Theta.*tmp; %\Pi_{v=1}^V * (...)
    end
    num_task = length(task_index)-1;
    S = zeros(N,1);
    for t= 1:num_task
        idx = task_index(t)+1:task_index(t+1);
        S(idx) = Pi_Z_Theta(idx,:) * Para.Phi(t,:)'; % F �ڶ���
        S(idx) = S(idx) + X(:,idx)' * Para.U(t,:)'; % F
    end    
end

function [S] = ComputeNorm(M, type)
    if strcmp(type,'L1')
        S = norm(M, 1);
    elseif strcmp(type,'L21')
        S = sum(sqrt(sum(abs(M).^2,2)));
    else % default is L2
        S = norm( M, 'fro');
    end    
end

function [Dv] = ComputeDv(M)
    eps = 1e-4;
    M_rnorm = sqrt(sum(abs(M).^2,2));
    Dv_diag = sqrt(eps + M_rnorm .* M_rnorm);
    Dv = bsxfun(@rdivide,M,Dv_diag);
end
