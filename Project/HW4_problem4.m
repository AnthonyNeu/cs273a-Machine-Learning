clc;
clear;
close all;

Ytrain = load('data/kaggle.Y.train.txt');
Xtrain = load('data/kaggle.X1.train.txt');
Xtest = load('data/kaggle.X1.test.txt');

[Xtr,Xte,Ytr,Yte] = splitData(Xtrain,Ytrain, .75);


% % train the regressor
Nbag = [1,5,10,15,20,25];
mseTrain = zeros(length(Nbag),1);
mseTest = zeros(length(Nbag),1);
[n,d] = size(Xtr);
for j = 1:length(Nbag)
    display(j);
    regressor = cell(1,Nbag(j));
    for i = 1: Nbag(j)
        [Xi,Yi] = bootstrapData(Xtr,Ytr,n);
        regressor{i} = treeRegress(Xi,Yi,'minParent',325,'maxDepth',8,'nFeatures',60);
    end

    %calculate the error
    [ntest,dtest] = size(Xte);
    predictTrain = zeros(Nbag(j),n);
    predictTest = zeros(Nbag(j),ntest);
    for i = 1: Nbag(j)
        predictTrain(i,:) = predict(regressor{i},Xtr);
        predictTest(i,:) = predict(regressor{i},Xte);
    end
    
    %calculate the mean of prediction
    if j > 1
        predictYr = (mean(predictTrain))';
        precictYe = (mean(predictTest))';
    else
        predictYr = predictTrain';
        precictYe = predictTest';
    end
    mseTrain(j) = mean( sum( (Ytr - predictYr).^2,2));
    mseTest(j) = mean( sum( (Yte - precictYe).^2,2));
end

%plot
plot(Nbag,mseTrain,'-ro',Nbag,mseTest,'-.b');
xlabel('number of learners');
ylabel('MSE');
legend('Train Error','Test Error');
title('influence of number of learners on decision tree');