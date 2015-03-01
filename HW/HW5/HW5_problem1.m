clc;
close all;
clear;

iris=load('data/iris.txt');     % load the text file
X = iris(:,1:2); Y=iris(:,end); % get first two features
XA = X(Y<2,:); YA=Y(Y<2);       % get class 0 vs 1

% %(a)
% figure;
% scatter(X(:,1),X(:,2),'filled');
% xlabel('feature 1');
% ylabel('feature 2');
% title('original data');
% 
% %(b)
% K = 5:15:20;
% scores = zeros(1,length(5:15:20));
% figure;
% for i=1:2
%     [assign,clusters,sumd] = kmeans(X,K(i));
%     crule = knnClassify( clusters, (1:K(i))', 1 );  
%     z = predict( crule, X );
%     subplot(2,1,i);
%     plotClassify2D([],X,z);
%     xlabel('feature 1');
%     ylabel('feature 2');
%     str = strcat('K =' ,num2str(K(i)),',clustering');
%     title(str);
%     scores(i) = sumd;
% end
% 
% scores1 = zeros(1,length(5:15:20));
% %random
% figure;
% for i=1:2
%     [assign,clusters,sumd] = kmeans(X,K(i),'random');
%     crule = knnClassify( clusters, (1:K(i))', 1 );  
%     z = predict( crule, X );
%     subplot(2,1,i);
%     plotClassify2D([],X,z);
%     xlabel('feature 1');
%     ylabel('feature 2');
%     str = strcat('K =' ,num2str(K(i)),',clustering,random');
%     title(str);
%     scores1(i) = sumd;
% end
% 
% %farthest
% figure;
% scores2 = zeros(1,length(5:15:20));
% for i=1:2
%     [assign,clusters,sumd] = kmeans(X,K(i),'farthest');
%     crule = knnClassify( clusters, (1:K(i))', 1 );  
%     z = predict( crule, X );
%     subplot(2,1,i);
%     plotClassify2D([],X,z);
%     xlabel('feature 1');
%     ylabel('feature 2');
%     str = strcat('K =' ,num2str(K(i)),',clustering,farthest');
%     title(str);
%     scores2(i) = sumd;
% end
% 
% %k++
% figure;
% scores3 = zeros(1,length(5:15:20));
% for i=1:2
%     [assign,clusters,sumd] = kmeans(X,K(i),'k++');
%     crule = knnClassify( clusters, (1:K(i))', 1 );  
%     z = predict( crule, X );
%     subplot(2,1,i);
%     plotClassify2D([],X,z);
%     xlabel('feature 1');
%     ylabel('feature 2');
%     str = strcat('K =' ,num2str(K(i)),',clustering,k++');
%     title(str);
%     scores3(i) = sumd;
% end
% 
% %(c)
% % single linkage
% Z = linkage(X,'single');
% T = cluster(Z,'maxclust',5);
% figure;
% subplot(2,1,1);
% plotClassify2D([],X,T);
% xlabel('feature 1');
% ylabel('feature 2');
% title('K=5,single linkage');
% 
% 
% Z = linkage(X,'single');
% T = cluster(Z,'maxclust',20);
% subplot(2,1,2);
% plotClassify2D([],X,T);
% xlabel('feature 1');
% ylabel('feature 2');
% title('K=20,single linkage');
% 
% 
% % complete
% Z = linkage(X,'complete');
% T = cluster(Z,'maxclust',5);
% figure;
% subplot(2,1,1);
% plotClassify2D([],X,T);
% xlabel('feature 1');
% ylabel('feature 2');
% title('K=5,complete linkage');
% 
% Z = linkage(X,'complete');
% T = cluster(Z,'maxclust',20);
% subplot(2,1,2);
% plotClassify2D([],X,T);
% xlabel('feature 1');
% ylabel('feature 2');
% title('K=20,complete linkage');

%(d)
K = 5:15:20;
figure;
for i=1:2
    [assign,clusters] = emCluster(X,K(i));
    crule = knnClassify( clusters.mu, (1:K(i))', 1 );  
    z = predict( crule, X );
    subplot(2,1,i);
    plotClassify2D([],X,z);
    xlabel('feature 1');
    ylabel('feature 2');
    str = strcat('K =' ,num2str(K(i)),',EM Gaussian');
    title(str);
end

%random
figure;
for i=1:2
    [assign,clusters] = emCluster(X,K(i),'random');
    crule = knnClassify( clusters.mu, (1:K(i))', 1 );  
    z = predict( crule, X );
    subplot(2,1,i);
    plotClassify2D([],X,z);
    xlabel('feature 1');
    ylabel('feature 2');
    str = strcat('K =' ,num2str(K(i)),',EM Gaussian,random');
    title(str);
end

%farthest
figure;
for i=1:2
    [assign,clusters] = emCluster(X,K(i),'farthest');
    crule = knnClassify( clusters.mu, (1:K(i))', 1 );  
    z = predict( crule, X );
    subplot(2,1,i);
    plotClassify2D([],X,z);
    xlabel('feature 1');
    ylabel('feature 2');
    str = strcat('K =' ,num2str(K(i)),',EM Gaussian,farthest');
    title(str);
end

%k++
figure;
for i=1:2
    [assign,clusters] = emCluster(X,K(i),'k++');
    crule = knnClassify( clusters.mu, (1:K(i))', 1 );  
    z = predict( crule, X );
    subplot(2,1,i);
    plotClassify2D([],X,z);
    xlabel('feature 1');
    ylabel('feature 2');
    str = strcat('K =' ,num2str(K(i)),',EM Gaussian,k++');
    title(str);
end

