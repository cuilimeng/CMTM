clear all
clc

load('nus_wide.mat')

for trail_num = 1:1
    
    rnd_tr = randperm(size(I_tr,1));
    rnd_te = randperm(size(I_te,1));
    
    T = [I_tr(rnd_tr<=220,:);I_te(rnd_te<=70,:)];
    I = [T_tr(rnd_tr<=220,:);T_te(rnd_te<=70,:)];
    LC = [L_tr(rnd_tr<=220,:);L_te(rnd_te<=70,:)];
    text_te_num = length(I_te(rnd_te<=70,1));
    
    CA = zeros(size(LC));
    for i = 1:size(LC,2)
        CA(LC(:,i)==1,i) = i;
    end        
    
    max_iid = length(I(:,1));
    max_tid = length(T(:,1));
    
    max_topic = 1;
    A = sparse((1:max_iid),(1:max_tid),1,max_iid,max_tid);
    
    for i = 1:max_iid
        tmp_idx1 = CA(i,CA(i,:)~=0);
        for j = 1:max_tid 
            tmp_idx2 = CA(j,CA(j,:)~=0);
            if any(ismember(tmp_idx1,tmp_idx2))
                A(i,j)=1;
            end
        end
    end
    C = ones(max_tid,max_topic);
    alpha = ones(max_topic,1) * 0.5;
    
    [U_1,U_2] = metric_learning (I,T,A);
    I = I * U_1;
    T = T * U_2;
    
    num_view = 6;
    % generes as topics
    train_data = cell(max_topic,1);
    test_data = cell(max_topic,1); 
    valid_data = cell(max_topic,1);
    
    train_label = cell(max_topic,1);
    test_label = cell(max_topic,1);
    valid_label = cell(max_topic,1);
    
    view_index = zeros(1,num_view);
    train_topic_num = zeros(1,max_topic);
    test_topic_num = zeros(1,max_topic);
    topic_testIdx = cell(max_topic,1);
    topic_trainIdx = cell(max_topic,1);
    test_idxs = [];
    
    for m = 1:max_topic
        text_idxs = find(C(:,m)>0); % 选出该topic对应的文本
        A_topic = A(:,text_idxs); % 选出该topic对应的label
        
        % 80% training 10% validation 10% testing
        image_num = size(A_topic,1);
        text_num_topic = length(text_idxs);
        
        trainNum = image_num * (text_num_topic - text_te_num);
        testNum = trainNum + image_num * text_te_num;
          
        trainIdx = (1:trainNum)';
        train_iids = rem(trainIdx-1,size(A_topic,1))+1;
        train_tids = text_idxs(floor((trainIdx-1)/size(A_topic,1))+1);
        topic_trainIdx{m} = sub2ind(size(A),train_iids,train_tids);
        
        testIdx = (trainNum+1:testNum)';
        test_iids = rem(testIdx-1,size(A_topic,1))+1;
        test_tids = text_idxs(floor((testIdx-1)/size(A_topic,1))+1);
        topic_testIdx{m} = sub2ind(size(A),test_iids,test_tids); 
        
        test_idxs = union(test_idxs, topic_testIdx{m});
    end
    
    for m = 1: max_topic
        topic_trainIdx{m} = setdiff(topic_trainIdx{m},test_idxs);
        N_train = length(topic_trainIdx{m});
        N_test = length(topic_testIdx{m});
        
        train_view = cell(num_view,1);
        test_view = cell(num_view,1);
        
        % view 1: image id
        train_iids = rem(topic_trainIdx{m}-1,size(A,1))+1;
        test_iids = rem(topic_testIdx{m}-1,size(A,1))+1;
        
        train_view{1} = sparse(train_iids,1:N_train,1,max_iid,N_train);
        test_view{1} = sparse(test_iids,1:N_test,1,max_iid,N_test);
                    
        % view 2: 
        train_view{2} = LC(train_iids,:)';
        test_view{2} = LC(test_iids,:)';
        
        % view 3: image feature
        train_view{3} = alpha(m) * I(train_iids,:)';
        test_view{3} = alpha(m) * I(test_iids,:)';
        
        % view 4: text id
        train_tids = floor((topic_trainIdx{m}-1)/size(A,1))+1;
        test_tids = floor((topic_testIdx{m}-1)/size(A,1))+1;
        
        train_view{4} = sparse(train_tids,1:N_train,1,max_tid,N_train);
        test_view{4} = sparse(test_tids,1:N_test,1,max_tid,N_test);
        
        % view 5: 
        train_view{5} = LC(train_tids,:)';
        test_view{5} = LC(test_tids,:)';
        
        % view 6: text feature  
        train_view{6} = (1-alpha(m)) * T(train_tids,:)';
        test_view{6} = (1-alpha(m)) * T(test_tids,:)';
        
        num_test = text_num_topic * round(length(topic_testIdx{m})/text_num_topic * 0.8);            
        train_label{m} = A(topic_trainIdx{m})';
        test_label{m} = A(topic_testIdx{m}(1:num_test))';
        valid_label{m} = A(topic_testIdx{m}(num_test+1:end))';
        train_data{m} = [];
        test_data{m} = [];
        
        for v = 1:num_view
            train_data{m} = [train_data{m}; train_view{v}];
            test_data{m} = [test_data{m}; test_view{v}(:,1:num_test)];
            valid_data{m} = [valid_data{m}; test_view{v}(:,num_test+1:end)];
        end
        
        if m==1
            last_dim = 0;
            for v = 1:num_view
                view_index(v) = last_dim + size(train_view{v},1);
                last_dim = view_index(v);
            end
        end
    end
    fname = ['demo-nus-it-' num2str(trail_num)];
    save(fname,'view_index','train_label','test_label','valid_label',...
        'train_data','test_data','valid_data','-v7.3');
    fprintf('=============\n');
end