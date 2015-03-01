clc;
close all;
clear;


data=load('data/curve80.txt');      % load the text file
y = data(:,2);                    % target value is last column
X = data(:,1);                % features are other columns
[Xtr,Xte,Ytr,Yte] = splitData(X,y, .75);  % split data into 75/25 train/test

%(b)
lr = linearRegress( Xtr, Ytr );  % create and train model
xs = [0:.05:10]'; % densely sample possible x-values
ys = predict( lr, xs ); % make predictions at xs

figure;
plot(xs,ys);
hold on;
scatter(Xtr,Ytr);

%training data error
MSETr = mse(lr,Xte,Yte);
display(MSETr);

%test data error
MSETe = mse(lr,Xtr,Ytr);
display(MSETe);

%(c)
d = [1,3,5,7,10,18];
errTrain = zeros(1,size(d,2));
errTest = zeros(1,size(d,2));
figure;
for n=1:size(d,2)
    XtrP = fpoly(Xtr, d(n), false); % create poly features up to given degree
    [XtrP, M,S] = rescale(XtrP); % it's often a good idea to scale the features 
    lr = linearRegress( XtrP, Ytr ); % create and train model
    XteP = rescale( fpoly(Xte,d(n),false), M,S);
    YhatTrain = predict( lr,XtrP); % predict on training data 
    YhatTest = predict( lr, XteP ); % predict on test data
    errTrain(n) = mse(lr,XtrP,Ytr);
    errTest(n) = mse(lr,XteP,Yte);
    
    %plot the f(x)
    xsP = rescale( fpoly(xs,d(n),false), M,S);
    ysP = predict( lr,xsP);
    subplot(2,3,n);
    plot(xs,ysP);
    title(strcat('d=',num2str(d(n))));
    if n==1
        ax = axis;
    end
    hold on;
    if n>1
        axis(ax);
    end
end

figure; 
title('Error Rate');
semilogy(d,errTrain,'-ro',d,errTest,'-.b');
legend('Training data error','Test data error');