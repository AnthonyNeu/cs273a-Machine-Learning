clc;
close all;
clear;

iris=load('data/iris.txt'); 
y=iris(:,end); 
X=iris(:,1:2);
[X,y] = shuffleData(X,y); % shuffle data randomly
% (This is a good idea in case your data are ordered in some pathological way,as the Iris data are)
[Xtr,Xte,Ytr,Yte] = splitData(X,y, .75); % split data into 75/25 train/test

%problem a
K = [1,5,10,50];
for i = 1: length(K)
    figure;
    knn = knnClassify( Xtr, Ytr, K(i) ); % replace or set K to some integer
    YteHat = predict( knn, Xte ); % make predictions on Xtest
    plotClassify2D( knn, Xtr, Ytr ); % make 2D classification plot with data (Xtr,Ytr)
end

%problem b
K=[1,2,5,10,50,100,200];
errTrain = zeros(1,length(K));
errTest = zeros(1,length(K));
for i=1:length(K)
    knn = knnClassify( Xtr, Ytr, K(i) ); % replace or set K to some integer
%     YtrHat = predict( knn, Xtr ); % make predictions on Xtrain
%     YHat = predict( knn, Xte );% make prediction on Xtest
%     %count the training error
%     for j=1:length(Ytr)
%         if(Ytr(j) ~= YtrHat(j))
%            errTrain(i) = errTrain(i) + 1;
%         end
%     end
%     %count the training error
%     for j=1:length(Yte)
%         if(Yte(j) ~= YHat(j))
%            errTest(i) = errTest(i) + 1;
%         end
%     end
%     errTrain(i) = errTrain(i)/length(Ytr);
%     errTest(i) = errTest(i)/length(Yte);
    errTrain(i) = err(knn,Xtr,Ytr);
    errTest(i) = err(knn,Xte,Yte);
end;
figure; 
title('Error Rate');
semilogx(K,errTrain,'-ro',K,errTest,'-.b');
legend('Training data error','Test data error');