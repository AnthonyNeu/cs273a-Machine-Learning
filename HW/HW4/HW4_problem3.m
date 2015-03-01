clc;
clear;
close all;

Ytrain = load('data/kaggle.Y.train.txt');
Xtrain = load('data/kaggle.X1.train.txt');
Xtest = load('data/kaggle.X1.test.txt');

[Xtr,Xte,Ytr,Yte] = splitData(Xtrain,Ytrain, .75);


%(a)
dt = treeRegress(Xtr,Ytr, 'maxDepth',20);
error1 = mse(dt,Xte,Yte);

%(b)
% errorTe = zeros(16,1);
% errorTr = zeros(16,1);
% for depth = 0: 15
%     dt = treeRegress(Xtr,Ytr, 'maxDepth',depth);    
%     errorTe(depth+1) = mse(dt,Xte,Yte);
%     errorTr(depth+1) = mse(dt,Xtr,Ytr);
% end
% 
% figure;
% title('influence of max depth on decision tree');
% plot(0:1:15,errorTe,'-ro',0:1:15,errorTr,'-.b');
% xlabel('Max Depth');
% ylabel('MSE');
% legend('Test Error','Train Error');


%(c)
% parent = 2.^[3:12];
% errorTe_parent = zeros(10,1);
% errorTr_parent = zeros(10,1);
% for n = 1: 10
%     dt = treeRegress(Xtr,Ytr, 'minParent',parent(n),'maxDepth',20);    
%     errorTe_parent(n) = mse(dt,Xte,Yte);
%     errorTr_parent(n) = mse(dt,Xtr,Ytr);
% end
% 
% figure;
% title('influence of min parent on decision tree');
% xlabel('Min Parent');
% ylabel('MSE');
% plot(parent,errorTe_parent,'-ro',parent,errorTr_parent,'-.b');
% legend('Test Error','Train Error');
% 
% 
% 
% %test
% parent = [400:10:500];
% errorTe_parent = zeros(11,1);
% errorTr_parent = zeros(11,1);
% for n = 1: 11
%     dt = treeRegress(Xtr,Ytr, 'minParent',parent(n),'maxDepth',8);    
%     errorTe_parent(n) = mse(dt,Xte,Yte);
%     errorTr_parent(n) = mse(dt,Xtr,Ytr);
% end
% 
% figure;
% title('influence of min parent on decision tree');
% xlabel('Min Parent');
% ylabel('MSE');
% plot(parent,errorTe_parent,'-ro',parent,errorTr_parent,'-.b');
% legend('Test Error','Train Error');
%     