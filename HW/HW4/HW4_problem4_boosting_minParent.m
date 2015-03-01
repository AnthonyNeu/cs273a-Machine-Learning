% clc;
% clear;
% close all;
% 
% Ytrain = load('data/kaggle.Y.train.txt');
% Xtrain = load('data/kaggle.X1.train.txt');
% Xtest = load('data/kaggle.X1.test.txt');
% 
% [Xtr,Xte,Ytr,Yte] = splitData(Xtrain,Ytrain, .75);
% 
% 
% % % train the regressor
% parent = [100:50:500];
% Nboost = 60;
% mseTrain = zeros(length(parent),1);
% mseTest = zeros(length(parent),1);
% [n,d] = size(Xtr);
% for j = 1:length(parent)
%     display(j);
%     mu = mean(Ytr);
%     dY = Ytr - mu;
%     learner = cell(1,Nboost);
%     alpha = ones(1,Nboost) * 0.5;
%     for i  = 1 : Nboost
%         learner{i} = treeRegress(Xtr,dY,'minParent',parent(j),'maxDepth',3); 
%         dY = dY - alpha(i) * predict(learner{i},Xtr);
%     end
%     
%     [ntest,dtest] = size(Xte);
%     [ntrain,dtrain] = size(Xtr);
%     predictTest = zeros(ntest,1);
%     predictTrain = zeros(ntrain,1);
%     for i  = 1 : Nboost
%         predictTest = predictTest + alpha(i) * predict(learner{i},Xte);
%         predictTrain = predictTrain + alpha(i) * predict(learner{i},Xtr);
%     end
%     mseTest(j) = mean(sum((Yte - predictTest - mu).^2,2));
%     mseTrain(j) = mean(sum((Ytr - predictTrain - mu).^2,2));
% end

%plot
xlabel('minParent');
ylabel('MSE');
plot(parent,mseTrain,'-ro',parent,mseTest,'-.b');
legend('Train Error','Test Error');
title('influence of minParent on decision tree using Boosting,alpha = 0.5');