clc;
clear;
close all;

Ytrain = load('data/kaggle.Y.train.txt');
Xtrain = load('data/kaggle.X1.train.txt');
Xtest = load('data/kaggle.X1.test.txt');

[Xtr,Xte,Ytr,Yte] = splitData(Xtrain,Ytrain, .75);

trn.X = Xtr;
trn.Y = Ytr;
tst.X = Xte;
tst.Y = Yte;

[trn,tst] = scaleSVM(trn,tst,0,1);



model = svmtrain(trn.Y, trn.X, '-s 3 -t 3 -h 0');

[Yhat, accuracy,label] = svmpredict(tst.Y, tst.X, model);

