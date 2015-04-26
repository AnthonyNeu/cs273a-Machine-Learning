clc;
clear;
close all;


Ytrain = load('/Users/anthony/Documents/Study Files/2015 Spring/CS273/Project/data/kaggle.Y.train.txt');
Xtrain = load('/Users/anthony/Documents/Study Files/2015 Spring/CS273/Project/data/kaggle.X1.train.txt');
Xtest = load('/Users/anthony/Documents/Study Files/2015 Spring/CS273/Project/data/kaggle.X1.test.txt');

[Xtr,Xte,Ytr,Yte] = splitData(Xtrain,Ytrain, .75);

model = svmtrain(Ytr, Xtr, '-s 3');


