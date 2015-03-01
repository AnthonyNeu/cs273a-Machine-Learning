clc;
clear;
close all;

iris=load('data/iris.txt');     % load the text file
X = iris(:,1:2); Y=iris(:,end); % get first two features
[X,Y] = shuffleData(X,Y);       % reorder randomly
X  = rescale(X);                % works much better for rescaled data
XA = X(Y<2,:); YA=Y(Y<2);       % get class 0 vs 1
XB = X(Y>0,:); YB=Y(Y>0);       % get class 1 vs 2


%(a)
figure;
scatter(XA(:,1),XA(:,2),50,YA,'filled');
title('class 0 vs class 1');
figure;
scatter(XB(:,1),XB(:,2),50,YB,'filled');
title('class 1 vs class 2');

%(b)
learnerA=logisticClassify2(); % create "blank" learner 
learnerA=setClasses(learnerA, unique(YA)); % define class labels using YA or YB
learnerB=logisticClassify2(); % create "blank" learner 
learnerB=setClasses(learnerB, unique(YB)); % define class labels using YA or YB
wts = [0.5 1 -0.25]; % TODO: fill in values 
learnerA=setWeights(learnerA, wts); % set the learner's parameters
learnerB=setWeights(learnerB,wts);
plot2DLinear(learnerA,XA,YA);
title('demo for XA');
plot2DLinear(learnerB,XB,YB);
title('demo for XB');

%(c)
YAte = predict(learnerA,XA);
YBte = predict(learnerB,XB);
errA = err(learnerA,XA,YA);
display(errA);
errB = err(learnerB,XB,YB);
display(errB);
%use plotClassify2D
figure;
plotClassify2D(learnerA,XA,YA);
figure;
plotClassify2D(learnerB,XB,YB);

%(e)
% learnA = logisticClassify2(XA,YA,'stepsize',0.5);
% figure;
% plotClassify2D(learnA,XA,YA);
% title('Train answer for XA');
% learnB = logisticClassify2(XB,YB,'stepsize',0.5);
% figure;
% plotClassify2D(learnB,XB,YB);
% title('Train answer for XB');