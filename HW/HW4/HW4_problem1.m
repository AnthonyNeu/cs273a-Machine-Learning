clc;
close all;
clear;

iris=load('data/iris.txt');     % load the text file
X = iris(:,1:2); Y=iris(:,end); % get first two features
XA = X(Y<2,:); YA=Y(Y<2);       % get class 0 vs 1

[n,d] = size(XA);
Xtest = [ones(n,1) XA];
H = zeros(d+1,d+1);
f = zeros(d+1,1);
% set the matrix H
for i= 2:d+1
   H(i,i) = 1;
end

%set the Diagonalization YA
YDiag = zeros(n,n);
for i = 1 : n
    if YA(i) == 0
        YDiag(i,i) = 1;
    end
    if YA(i) == 1
        YDiag(i,i) = -1;
    end
end

 %set the A,b
A = YDiag * Xtest;
b = -1 *ones(n,1);

%solve the quadric programming
theta = quadprog(H,f,A,b);

% use plot2DLinear
learnerA=logisticClassify(); % create "blank" learner 
learnerA=setClasses(learnerA, unique(YA)); % define class labels using YA
learnerA=setWeights(learnerA, theta'); % set the learner's parameters
plotClassify2D(learnerA,XA,YA);
title('Decision boundary for XA');


