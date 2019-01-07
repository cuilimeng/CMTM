% tuning parameters
function [bestPara] = cmtm_grid(data_id, num_metric)
    base = 10; 
    k = 20;    
    n = 3; 
    m = 8;
    para.IsCv = 1;
    
    result_avg = cell(num_metric,1);
    for metric = 1:num_metric
        result_avg{metric} = cell(n+1,m+1);
        for i = 1 : n+1
            for j = 1 : m+1
                result_avg{metric}{ i, j } = Inf;
            end;
        end;
    end
    para1 = base.^(-5+(1:n)); % Identify different parameters
    para2 = base.^(-4+(1:m));
    for metric = 1:num_metric % The grid
        for i = 1 : n
            result_avg{metric}{ i+1, 1 } = para1(i);
        end
        for j = 1 : m
            result_avg{metric}{ 1, j+1 } = para2(j);
        end
    end
    % Calculate the evaluation result of each parameter combination
    for j = 1 : m
        for i = 1:n
            para.para1 =  para1(i);
            para.para2 = para2(j);
            para.k = k;
            [ avg_metrics, ~ ] = exp_func(data_id, para);
             for metric = 1:num_metric
                result_avg{metric}{ i+1, j+1 } = avg_metrics(metric);
                csvwrite( [ 'result/avg-cmtf-' data_id '-m' num2str(metric) '.csv' ],...
                    result_avg{metric} );
             end            
        end;
    end;
    % Identify the optimal parameters and evaluation results
    bestPara = cell(metric,1);
    for metric = 1:num_metric 
        [~,idx] = max([result_avg{metric}{2:end,2:end}]);
        idx1 = rem(idx-1, size(result_avg{metric},1)-1)+1;
        bestPara{metric}.para1 = para1(idx1);
        idx2 = floor((idx-1)/ (size(result_avg{metric},1)-1))+1;
        bestPara{metric}.para2 = para2(idx2);
    end
end


