classdef slexpevalMAP< slexpeval
    % Detailed explanation goes here
    
    properties
    end
    
    methods
      function s = slexpevalMAP()
            s = s@slexpeval('MAP','mean-average-precision');
            s.discription='MAP';
      end
      function [value,s] = evaluate(s,labels,pre_labels,view_index)
          num_task = size(labels,1);
          values = zeros(num_task,1);
          for t = 1:num_task
              %
              N_base = view_index(1);
              Q = length(pre_labels{t})/N_base;
              rat = 10;
              R = round((rat * N_base)/100);
              
              pre_labels_q = reshape(pre_labels{t},N_base,Q);
              labels_q = reshape(labels{t},N_base,Q);
              
              [~,ind] = sort(pre_labels_q,1,'descend');
              
              P_q = zeros(R,Q);
              AP = zeros(Q,1);
              for q = 1:Q
                  tmp_count = 1;
                  for r = 1:R
                      if labels_q(ind(r,q),q) == 1
                          P_q(r,q) = tmp_count/r;
                          tmp_count = tmp_count +1;
                      end                      
                  end
                  AP(q) = sum(P_q(:,q))/(tmp_count - 1+1e-10);
              end
              values(t) = sum(AP)/Q; %mAP
          end
          value = mean(values);
          s.value=[s.value value];
      end
    end
    
end