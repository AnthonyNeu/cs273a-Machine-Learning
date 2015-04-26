clc;
clear;
close all;

Ytrain = load('data/kaggle.Y.train.txt');
Xtrain = load('data/kaggle.X1.train.txt');
Xtest = load('data/kaggle.X1.test.txt');

[Xtr,Xte,Ytr,Yte] = splitData(Xtrain,Ytrain, .75);

%minParent
% parent = 2.^[3:12];
% errorTe_parent = zeros(10,1);
% errorTr_parent = zeros(10,1);
% for n = 1: 10
%     dt = treeRegress(Xtr,Ytr, 'minParent',parent(n),'maxDepth',9);    
%     errorTe_parent(n) = mse(dt,Xte,Yte);
%     errorTr_parent(n) = mse(dt,Xtr,Ytr);
% end
% 
% figure;
% xlabel('Min Parent');
% ylabel('MSE');
% plot(parent,errorTe_parent,'-ro',parent,errorTr_parent,'-.b');
% legend('Test Error','Train Error');
% title('influence of min parent on decision tree,maxDepth = 9');


%minParent
% parent = [200:10:500];
% errorTe_parent = zeros(31,1);
% errorTr_parent = zeros(31,1);
% for n = 1: 31
%     dt = treeRegress(Xtr,Ytr, 'minParent',parent(n),'maxDepth',8);    
%     errorTe_parent(n) = mse(dt,Xte,Yte);
%     errorTr_parent(n) = mse(dt,Xtr,Ytr);
% end
% 
% figure;
% xlabel('Min Parent');
% ylabel('MSE');
% plot(parent,errorTe_parent,'-ro',parent,errorTr_parent,'-.b');
% legend('Test Error','Train Error');
% title('influence of min parent on decision tree,maxDepth = 8');

% %upload the test
dt = treeRegress(Xtr,Ytr, 'minParent',325,'maxDepth',8);
Ytest = predict(dt,Xtest);
fh = fopen('predictions.csv','w');  % open file for upload
fprintf(fh,'ID,Prediction\n');      % output header line
for i=1:length(Ytest),
    fprintf(fh,'%d,%d\n',i,Ytest(i));  % output each prediction
end;
fclose(fh);                         % close the file

