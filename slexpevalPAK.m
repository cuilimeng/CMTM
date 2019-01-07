classdef slexpevalPAK< slexpeval
    % Detailed explanation goes here
    
    properties
    end
    
    methods
      function s = slexpevalPAK()
            s = s@slexpeval('P@K','Precision@K');
            s.discription='P@K';
      end
      function [value,s] = evaluate(s,labels,pre_labels,view_index)
          num_task = size(labels,1);
          values = zeros(num_task,1);
          for t = 1:num_task
              %
              N_base = view_index(1);
              Q = length(pre_labels{t})/N_base;
              K = 100;
              
              pre_labels_q = reshape(pre_labels{t},N_base,Q);
              labels_q = reshape(labels{t},N_base,Q);

              [~,ind] = sort(pre_labels_q,1,'descend');
              
              P_q = zeros(Q,1);
              for q = 1:Q
                  tmp_count = 0;
                  for r = 1:K
                      if labels_q(ind(r,q),q) == 1                          
                          tmp_count = tmp_count +1;
                      end                      
                  end
                  P_q(q) = tmp_count/K;
              end			  
              values(t) = mean(P_q); %Precision@100
          end
          value = mean(values);
          s.value=[s.value value];
      end
    end
    
end