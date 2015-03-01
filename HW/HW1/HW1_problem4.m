clc;
close all;
clear;

iris=load('data/iris.txt'); 
y=iris(:,end); 
X=iris(:,1:2);
[X,y] = shuffleData(X,y); % shuffle data randomly
[Xtr,Xte,Ytr,Yte] = splitData(X,y, .75); % split data into 75/25 train/test

%(a)
%count the number of points in each class
sum1 = 0;
sum2 = 0;
sum3 = 0;
for i = 1:size(Xtr,1)
   if(y(i)==0)
       sum1 = sum1 + 1;
   end
   if(y(i)==1)
       sum2 = sum2 + 1;
   end
   if(y(i)==2)
       sum3 = sum3 + 1;
   end
end

%get the point in each class
X1 = zeros(sum1,2);
X2 = zeros(sum2,2);
X3 = zeros(sum3,2);
sum1 = 1;
sum2 = 1;
sum3 = 1;
for i = 1:size(Xtr,1)
   if(y(i)==0)
       X1(sum1,:) = Xtr(i,:);
       sum1 = sum1 + 1;
   end
   if(y(i)==1)
       X2(sum2,:) = Xtr(i,:);
       sum2 = sum2 + 1;
   end
   if(y(i)==2)
       X3(sum3,:) = Xtr(i,:);
       sum3 = sum3 + 1;
   end
end

%calculate the mean and covriance for each class
means = zeros(3,2);
covriance = zeros(3*2,2);
means(1,:) = mean(X1);
means(2,:) = mean(X2);
means(3,:) = mean(X3);
covriance(1:2,:) = cov(X1);
covriance(3:4,:) = cov(X2);
covriance(5:6,:) = cov(X3);

%(b)
Color = zeros(size(Xtr,1),3);
for i = 1 : size(Xtr,1)
   if (y(i)==0)
       Color(i,:) = [0 0 0];%black
   end
   if (y(i)==1)
       Color(i,:) = [1 0 0];%red
   end
   if (y(i)==2)
       Color(i,:) = [0 1 1];%blue
   end
end
figure;
scatter(Xtr(:,1),Xtr(:,2),50,Color,'filled');

%(c)
hold on;
plotGauss2D(means(1,:),covriance(1:2,:),'k');
plotGauss2D(means(2,:),covriance(3:4,:),'r');
plotGauss2D(means(3,:),covriance(5:6,:),'b');

%(d)
figure;
bc = gaussBayesClassify( Xtr, Ytr );
plotClassify2D(bc, Xtr, Ytr);
errTrain = err(bc,Xtr,Ytr);
errTest = err(bc,Xte,Yte);

%(f)
iris=load('data/iris.txt'); 
y=iris(:,end); 
X=iris(:,1:end-1);
[X,y] = shuffleData(X,y); % shuffle data randomly
[Xtr,Xte,Ytr,Yte] = splitData(X,y, .75); % split data into 75/25 train/test
bc = gaussBayesClassify( Xtr, Ytr );
errTrain1 = err(bc,Xtr,Ytr);
errTest1 = err(bc,Xte,Yte);


