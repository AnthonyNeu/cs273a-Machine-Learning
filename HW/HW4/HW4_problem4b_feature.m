% clc;
% clear;
% close all;
% 
% Ytrain = load('data/kaggle.Y.train.txt');
% Xtrain = load('data/kaggle.X1.train.txt');
% Xtest = load('data/kaggle.X1.test.txt');
% 
% [Xtr,Xte,Ytr,Yte] = splitData(Xtrain,Ytrain, .75);
% 
% [ntr,d] = size(Xtr);
% features = 10:10:90;
% Nbag = 25;
% errorTrain = zeros(length(features),1);
% errorTest = zeros(length(features),1);
% 
% for n = 1 : length(features)
%     regressor = cell(1,Nbag);
%     for i = 1: Nbag
%         [Xi,Yi] = bootstrapData(Xtr,Ytr,ntr);
%         regressor{i} = treeRegress(Xi,Yi,'minParent',256,'maxDepth',15,'nFeatures',features(n));
%     end
%     [ntest,dtest] = size(Xte);
%     predictTrain = zeros(Nbag,ntr);
%     predictTest = zeros(Nbag,ntest);
%     for i = 1 : Nbag
%         predictTrain(i,:) = predict(regressor{i},Xtr);
%         predictTest(i,:) = predict(regressor{i},Xte);
%     end
%     errorTrain(n) = mean( sum( (Ytr - (mean(predictTrain))').^2,2));
%     errorTest(n) = mean( sum( (Yte - (mean(predictTest))').^2,2));
% end   


% %plot
% figure;
% xlabel('number of features');
% ylabel('MSE');
% plot(features,errorTest,'-ro',features,errorTrain,'-.b');
% legend('Test Error','Train Error');
% title('influence of number of features on decision tree with Bag = 25');


