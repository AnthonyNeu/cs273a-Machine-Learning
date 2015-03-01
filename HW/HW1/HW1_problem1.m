clc;
close all;
clear;

iris=load('data/iris.txt');     % load the text file
y = iris(:,end);           % target value is last column
X = iris(:,1:end-1);       % features are other columns
whos % show current variables in memory and sizes
NumofFeatures = size(X,2);
NumofPoints = size(X,1);

%For each feature, plot a histogram ("hist") of the data values
for i = 1:NumofFeatures,
    figure;
    xlabel('value of feature');
    ylabel('#of points');
    hist(iris(:,i));
end;

%Compute the mean of the data points for each feature 
means = zeros(1,NumofFeatures);
for i = 1:NumofFeatures,
    means(1,i) = mean(iris(:,i));
end;

%Compute the variance and standard deviation of the data points for each feature
variance = zeros(1,NumofFeatures);
stde = zeros(1,NumofFeatures);
for i = 1:NumofFeatures,
    variance(1,i) = var(iris(:,i));
    stde(1,i) = sqrt(variance(1,i));
end;

%Normalize the data by subtracting the mean value from each feature, 
%and dividing by its standard deviation.
X_normalize = X;
for i = 1:NumofFeatures,
    X_normalize(:,i) = (X(:,i)-means(i))/(stde(i));
end;

%For each pair of features, plot a scatterplot
Color = zeros(NumofPoints,3);
for i = 1 : NumofPoints
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
%pair(1,2)
figure;
scatter(iris(:,1),iris(:,2),50,Color,'filled');
%pair(1,3)
figure;
scatter(iris(:,1),iris(:,3),50,Color,'filled');
%pair(1,4)
figure;
scatter(iris(:,1),iris(:,4),50,Color,'filled');