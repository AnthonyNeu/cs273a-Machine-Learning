clc;
clear;
close all;

Ytrain = load('data/kaggle.Y.train.txt');
Xtrain = load('data/kaggle.X1.train.txt');
Xtest = load('data/kaggle.X1.test.txt');

[Xtr,Xte,Ytr,Yte] = splitData(Xtrain,Ytrain, .75);


% % train the regressor
Nboost = [1,5,10,15,20,25,30,35,40,45,50,60];
mseTrain = zeros(length(Nboost),1);
mseTest = zeros(length(Nboost),1);
[n,d] = size(Xtr);
for j = 1:length(Nboost)
    display(j);
    mu = mean(Ytr);
    dY = Ytr - mu;
    learner = cell(1,Nboost(j));
    alpha = ones(1,Nboost(j)) * 0.5;
    for i  = 1 : Nboost(j)
        learner{i} = treeRegress(Xtr,dY,'minParent',200,'maxDepth',3); 
        dY = dY - alpha(i) * predict(learner{i},Xtr);
    end
    
    [ntest,dtest] = size(Xte);
    [ntrain,dtrain] = size(Xtr);
    predictTest = zeros(ntest,1);
    predictTrain = zeros(ntrain,1);
    for i  = 1 : Nboost(j)
        predictTest = predictTest + alpha(i) * predict(learner{i},Xte);
        predictTrain = predictTrain + alpha(i) * predict(learner{i},Xtr);
    end
    mseTest(j) = mean(sum((Yte - predictTest - mu).^2,2));
    mseTrain(j) = mean(sum((Ytr - predictTrain - mu).^2,2));
end

%plot
xlabel('number of learners');
ylabel('MSE');
plot(Nboost,mseTrain,'-ro',Nboost,mseTest,'-.b');
legend('Train Error','Test Error');
title('influence of number of learners on decision tree using Boosting,alpha = 0.25');