clc;
close all;
clear;

data=load('data/curve80.txt');      % load the text file
y = data(:,2);                    % target value is last column
X = data(:,1);                % features are other columns
[Xtr,Xte,Ytr,Yte] = splitData(X,y, .75);  % split data into 75/25 train/test


d = [1,3,5,7,10,18];
err = zeros(1,size(d,2));
for n=1:size(d,2)
    nFolds = 5;
    J = zeros(1,nFolds);
    for iFold = 1:nFolds,
        [Xti,Xvi,Yti,Yvi] = crossValidate(Xtr,Ytr,nFolds,iFold); % take ith data block as v
        XtiP = fpoly(Xti, d(n), false);
        [XtiP, M,S] = rescale(XtiP);
        learner = linearRegress(XtiP,Yti);
        J(iFold) = mse(learner,rescale(fpoly(Xvi,d(n),false), M,S),Yvi);
    end
    err(n) = mean(J);
end

figure; 
title('Error Rate');
semilogy(d,err,'-ro');
hold on;

d = [1,3,5,7,10,18];
errTrain = zeros(1,size(d,2));
errTest = zeros(1,size(d,2));
for n=1:size(d,2)
    XtrP = fpoly(Xtr, d(n), false); % create poly features up to given degree
    [XtrP, M,S] = rescale(XtrP); % it's often a good idea to scale the features 
    lr = linearRegress( XtrP, Ytr ); % create and train model
    XteP = rescale( fpoly(Xte,d(n),false), M,S);
    YhatTrain = predict( lr,XtrP); % predict on training data 
    YhatTest = predict( lr, XteP ); % predict on test data
    errTrain(n) = mse(lr,XtrP,Ytr);
    errTest(n) = mse(lr,XteP,Yte);
end
semilogy(d,errTest,'-.b');
legend('cross-validation','actual test data');
