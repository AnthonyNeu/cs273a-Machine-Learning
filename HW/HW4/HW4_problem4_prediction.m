clc;
clear;
close all;

Ytrain = load('data/kaggle.Y.train.txt');
Xtrain = load('data/kaggle.X1.train.txt');
Xtest = load('data/kaggle.X1.test.txt');

[Xtr,Xte,Ytr,Yte] = splitData(Xtrain,Ytrain, .75);
[ntr,d] = size(Xtr);
Nbag = 25;
regressor = cell(1,Nbag);
mid = 40;
for i = 1: Nbag
    [Xi,Yi] = bootstrapData(Xtr,Ytr,ntr);
    regressor{i} = treeRegress(Xi,Yi,'minParent',50,'maxDepth',20,'nFeatures',mid + i);
end
[ntest,dtest] = size(Xtest);
predictTest = zeros(Nbag,ntest);
for i = 1 : Nbag
    predictTest(i,:) = predict(regressor{i},Xtest);
end

%upload the test
Ytest = (mean(predictTest))';
fh = fopen('predictions.csv','w');  % open file for upload
fprintf(fh,'ID,Prediction\n');      % output header line
for i=1:length(Ytest),
    fprintf(fh,'%d,%d\n',i,Ytest(i));  % output each prediction
end;
fclose(fh);                         % close the file

