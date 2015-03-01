clc;
clear;
close all;

Ytrain = load('data/kaggle.Y.train.txt');
Xtrain = load('data/kaggle.X1.train.txt');
Xtest = load('data/kaggle.X1.test.txt');
[Xtr,Xte,Ytr,Yte] = splitData(Xtrain,Ytrain, .75);
%prediction
    Nboost = 1000;
    mu = mean(Ytrain);
    dY = Ytrain - mu;
    learner = cell(1,Nboost);
    alpha = ones(1,Nboost) * 0.01;
    for i  = 1 : Nboost
        display(i);
        learner{i} = treeRegress(Xtrain,dY,'minParent',200,'maxDepth',25); 
        dY = dY - alpha(i) * predict(learner{i},Xtrain);
    end
    
    [ntest,dtest] = size(Xtest);
    Ytest = zeros(ntest,1);
    for i  = 1 : Nboost
        Ytest = Ytest + alpha(i) * predict(learner{i},Xtest);
    end
Ytest = Ytest + mu;
%upload the test
fh = fopen('predictions.csv','w');  % open file for upload
fprintf(fh,'ID,Prediction\n');      % output header line
for i=1:length(Ytest),
    fprintf(fh,'%d,%d\n',i,Ytest(i));  % output each prediction
end;
fclose(fh);                         % close the file