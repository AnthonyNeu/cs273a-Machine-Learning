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
% trn.X = Xtr;
% trn.Y = Ytr;
% tst.X = Xte;
% tst.Y = Yte;
% 
% [trn,tst] = scaleSVM(trn,tst,0,1);

%% test for gamma in polynomial
% gamma = [0.0001,0.0005,0.001,0.005,0.01,0.005,0.01,0.05,0.1 ];
% mseTrain = zeros(1,length(gamma));
% mseTest = zeros(1,length(gamma));
% 
% parfor i = 1: length(gamma)
%     i
%     s = ['-s 3 -t 1 -h 0 -g ',num2str(gamma(i))];
%     model = svmtrain(trn.Y, trn.X, s);
%     [Yhat, accuracy,label] = svmpredict(trn.Y, trn.X, model);
%     mseTrain(i) = accuracy(2);
%     [Yhat, accuracy,label] = svmpredict(tst.Y, tst.X, model);
%     mseTest(i) = accuracy(2);
% end

% %plot
% figure;
% plot(gamma,mseTrain,'-ro',gamma,mseTest,'-.b');
% xlabel('gamma');
% ylabel('MSE');
% legend('Train Error','Test Error');
% title('influence of gamma on libsvm with polynomial kernel');


% % test for C in polynomial
% C = [1:5:51];
% mseTrain = zeros(1,length(C));
% mseTest = zeros(1,length(C));
% 
% parfor i = 1: length(C)
%     i
%     s = ['-s 3 -t 1 -h 0 -c ',num2str(C(i))];
%     model = svmtrain(trn.Y, trn.X, s);
%     [Yhat, accuracy,label] = svmpredict(trn.Y, trn.X, model);
%     mseTrain(i) = accuracy(2);
%     [Yhat, accuracy,label] = svmpredict(tst.Y, tst.X, model);
%     mseTest(i) = accuracy(2);
% end

%plot
figure;
plot(C,mseTrain,'-ro',C,mseTest,'-.b');
xlabel('C');
ylabel('MSE');
legend('Train Error','Test Error');
title('influence of cost C on libsvm with polynomial kernel');

% test for degree in polynomial
% degree = [1:1:10];
% mseTrain = zeros(1,length(degree));
% mseTest = zeros(1,length(degree));
% parfor i = 1: length(degree)
%     i
%     s = ['-s 3 -t 1 -h 0 -d ',num2str(degree(i))];
%     model = svmtrain(trn.Y, trn.X, s);
%     [Yhat, accuracy,label] = svmpredict(trn.Y, trn.X, model);
%     mseTrain(i) = accuracy(2);
%     [Yhat, accuracy,label] = svmpredict(tst.Y, tst.X, model);
%     mseTest(i) = accuracy(2);
% end
% 
% %plot
% figure;
% plot(degree,mseTrain,'-ro',degree,mseTest,'-.b');
% xlabel('degree');
% ylabel('MSE');
% legend('Train Error','Test Error');
% title('influence of degree on libsvm with polynomial kernel');


