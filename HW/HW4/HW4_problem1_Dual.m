clc;
close all;
clear;

iris=load('data/iris.txt');     % load the text file
X = iris(:,1:2); Y=iris(:,end); % get first two features
XA = X(Y<2,:); YA=Y(Y<2);       % get class 0 vs 1

[n,d] = size(XA);
f = -1 * ones(n,1);
Ytest = zeros(n,1);
for i = 1: n
    if YA(i) == 0
        Ytest(i) = -1;
    end
    if YA(i) == 1
        Ytest(i) = 1;
    end
end
H = zeros(n,n);
% calculate H
for i = 1: n
    for j = 1: n
        H(i,j) = Ytest(i) * Ytest(j) * (XA(i,:) * XA(j,:)');
    end
end

% Aeq,beq
Aeq = Ytest';
beq = 0;

% A,b
A = [];
b = [];

% lb,ub
lb = zeros(n,1);
ub = inf * ones(n,1);

%calculate the alpha
alpha = quadprog(H,f,A,b,Aeq,beq,lb,ub);

%calculate the omiga
omiga = XA' * (alpha .* Ytest);

% set the eplison
eplison = 0.5;
% find the support vector
sv = alpha > eplison;

% calculate the b
b = 1/sum(sv) * sum(sv' *(Ytest-XA*omiga));

% use plot2DLinear
learnerA=logisticClassify(); % create "blank" learner 
learnerA=setClasses(learnerA, unique(YA)); % define class labels using YA
learnerA=setWeights(learnerA, [b omiga']); % set the learner's parameters
plotClassify2D(learnerA,XA,YA);
title('Decision boundary for XA');

% plot the support vector
hold on;
for i = 1: n
   if sv(i) == 1
       scatter(XA(i,1),XA(i,2),50,'g','filled')
   end
end    
