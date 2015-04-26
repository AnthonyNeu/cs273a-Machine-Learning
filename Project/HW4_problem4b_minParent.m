clc;
clear;
close all;

Ytrain = load('data/kaggle.Y.train.txt');
Xtrain = load('data/kaggle.X1.train.txt');
Xtest = load('data/kaggle.X1.test.txt');

[Xtr,Xte,Ytr,Yte] = splitData(Xtrain,Ytrain, .75);

[ntr,d] = size(Xtr);
parent = [10:50:510];
Nbag = 25;
errorTrain = zeros(length(parent),1);
errorTest = zeros(length(parent),1);

for n = 1 : length(parent)
    display(n);
    regressor = cell(1,Nbag);
    for i = 1: Nbag
        [Xi,Yi] = bootstrapData(Xtr,Ytr,ntr);
        regressor{i} = treeRegress(Xi,Yi,'minParent',parent(n),'maxDepth',15,'nFeatures',60);
    end
    [ntest,dtest] = size(Xte);
    predictTrain = zeros(Nbag,ntr);
    predictTest = zeros(Nbag,ntest);
    for i = 1 : Nbag
        predictTrain(i,:) = predict(regressor{i},Xtr);
        predictTest(i,:) = predict(regressor{i},Xte);
    end
    errorTrain(n) = mean( sum( (Ytr - (mean(predictTrain))').^2,2));
    errorTest(n) = mean( sum( (Yte - (mean(predictTest))').^2,2));
end   


%plot
figure;
xlabel('minParent');
ylabel('MSE');
plot(parent,errorTest,'-ro',parent,errorTrain,'-.b');
legend('Test Error','Train Error');
title('influence of minParent on decision tree with Bag = 25');