clc;
clear;
close all;

Ytrain = load('data/kaggle.Y.train.txt');
Xtrain = load('data/kaggle.X1.train.txt');
Xtest = load('data/kaggle.X1.test.txt');

[Xtr,Xte,Ytr,Yte] = splitData(Xtrain,Ytrain, .75);

trn.X = Xtr;
trn.Y = Ytr;
tst.X = Xtest;

[trn,tst] = scaleSVM(trn,tst,0,1);

model = svmtrain(trn.Y, trn.X, '-s 3 -g 0.05 -c 5 -h 0');

[Yhat, accuracy,label] = svmpredict(zeros(40000,1), tst.X, model);


%upload the test
fh = fopen('predictions1.csv','w');  % open file for upload
fprintf(fh,'ID,Prediction\n');      % output header line
for i=1:length(Yhat),
    fprintf(fh,'%d,%d\n',i,Yhat(i));  % output each prediction
end;
fclose(fh);                         % close the file

