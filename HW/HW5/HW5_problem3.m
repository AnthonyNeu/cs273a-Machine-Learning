clc;
close all;
clear;

X = load('data/faces.txt');
[n,d] = size(X);
for i = 1 : n
    img = reshape(X(i,:),[24 24]);
end
%imagesc(img); axis square; colormap gray;

%(a)
mu = mean(X,1);
X0 = X - repmat(mu,[n,1]); 

%(b)
[U,S,V] = svds(X0,10);

%(c)
W = U*S;
mse = zeros(10,1);
for k = 1:10
    X0hat = W(:,1:k)* V(:,1:k)';
    mse(k) = mean(mean(X0-X0hat).^2);
end
figure;
plot(1:10,mse);
xlabel('K');
ylabel('mse');

%(d)
% X1 = repmat(mu,[10,1]);
% for k = 1 : 10
%     alpha = median(abs(W(:,k)));
%     X1(k,:) = X1(k,:) + alpha * V(:,k)';
% end
% 
% figure;
% for k = 1 : 10
%     img = reshape(X1(k,:),[24 24]);
%     subplot(5,2,k);
%     imagesc(img); axis square; colormap gray;
%     str = strcat('K=',num2str(k));
%     title(str);
% end

%(e)
% idx = [15:1:25]; % pick some data at random or otherwise
% figure; hold on; axis ij; colormap(gray);
% range = max(W(idx,1:2)) - min(W(idx,1:2)); % find range of coordinates to be plot 
% scale = [200 200]./range; % want 24x24 to be visible but not lar
% for i=idx, imagesc(W(i,1)*scale(1),W(i,2)*scale(2), reshape(X(i,:),24,24)); end;

%(f)
%pick the 10th and 15th
%k=5
[U,S,V] = svds(X0,5);
W = U * S;
%rebuild the 10th
X10 = W(10,:)* V';
img = reshape(X10,[24 24]);
figure;
imagesc(img); axis square; colormap gray;
title('10th,k=5');

X15 = W(15,:)* V';
img = reshape(X15,[24 24]);
figure;
imagesc(img); axis square; colormap gray;
title('15th,k=5');

%k=10
[U,S,V] = svds(X0,10);
W = U * S;
%rebuild the 10th
X10 = W(10,:)* V';
img = reshape(X10,[24 24]);
figure;
imagesc(img); axis square; colormap gray;
title('10th,k=10');

X15 = W(15,:)* V';
img = reshape(X15,[24 24]);
figure;
imagesc(img); axis square; colormap gray;
title('15th,k=10');

%k=50
[U,S,V] = svds(X0,50);
W = U * S;
%rebuild the 10th
X10 = W(10,:)* V';
img = reshape(X10,[24 24]);
figure;
imagesc(img); axis square; colormap gray;
title('10th,k=50');

X15 = W(15,:)* V';
img = reshape(X15,[24 24]);
figure;
imagesc(img); axis square; colormap gray;
title('15th,k=50');

