clear all
close all
clc

trial_num = 1;
num_metric = 2; % The number of evaluation indicators: 1.Precision@100; 2.MAP;
result_avg = cell(num_metric,1);
result_std = cell(num_metric,1);
%Initialize result_avg and result_std
for metric = 1:num_metric 
    result_avg{metric} = cell(1,trial_num);
    result_std{metric} = cell(1,trial_num);    
    for j = 1 : trial_num
        result_avg{metric}{ 1, j } = Inf;
        result_std{metric}{ 1, j } = Inf;
    end;   
end

% Initialize the optimal parameter matrix
bestPara = cell(1,1);
for i = 1:trial_num
    data_id = 'demo-nus-ti';
    fname = [data_id '-' num2str(i)];
    k = 20; % The factor number of CP factorization
    para.IsCv = 0;
    
    tmpPara = cmtm_grid(fname, num_metric); % Grid search
    bestPara = tmpPara{end}; % Identify the best parameters
    para.para1 = bestPara.para1;
    para.para2 = bestPara.para2;
    para.k = k;
    [avg_metrics,std_metrics] = exp_func(fname, para);
    
    for metric = 1:num_metric
        result_avg{metric}{ 1, i} = avg_metrics(metric);
        csvwrite( [ 'result/' data_id '-m' num2str(metric) '.csv' ], result_avg{metric});
    end
end