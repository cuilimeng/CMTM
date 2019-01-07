function [exp_avg,exp_std] = exp_func(data_id,para)
    num_iter = 200;
    if para.IsCv == 1
       dataset=slexpdatasetTrainValid();
       para.IsCv = 0;
%        dataset=slexpdatasetTrainOnly();
    else
        dataset=slexpdatasetTrainTest();
    end
    dataset.dset=data_id;
    if ~exist('result','dir')
        mkdir('result');
    end
    if ~exist(['result/' dataset.name],'dir')
        mkdir('result/',dataset.name);
    end
    if ~exist('tmp','dir')
        mkdir('tmp');
    end
    
    evalmethods{1}=slexpevalPAK();
    evalmethods{2}=slexpevalMAP();
%       
    %%     
    cla=slexpclassifierMFM();
    expsetting=slexpSettingMVMT();
    disp(['Running ====='  dataset.name '---' cla.name '-----']);
    result=['result/' dataset.name '/' cla.name '-' data_id '.mat'];
    disp(result);
    exp=slexprofile(dataset,expsetting,cla,evalmethods,result);
    exp.classifier.lambda_ = para.para1;
    exp.classifier.gamma1_ = para.para2;
    exp.classifier.regular_task = 'L2';
    exp.classifier.regular_view = 'L2';
    exp.classifier.regular_u = 'L21';
    exp.classifier.k_= para.k;
    exp.classifier.iteration_=num_iter;
    exp=exp.run(para.IsCv);
        
    numEval = size(exp.expsetting.evaluations(:),1);
    exp_avg = zeros(numEval,1);
    exp_std = zeros(numEval,1);
    for i=1:numEval
        exp_avg(i) = mean(exp.expsetting.evaluations{i}.value);
        exp_std(i) = std(exp.expsetting.evaluations{i}.value);
    end;
end
