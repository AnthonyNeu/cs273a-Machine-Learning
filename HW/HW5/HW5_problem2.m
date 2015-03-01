clc;
close all;
clear;

% Read in vocabulary and data (word counts per document)
[vocab] = textread('data/text/vocab.txt','%s');
[did,wid,cnt] = textread('data/text/docword.txt','%d%d%d','headerlines',3);
X = sparse(did,wid,cnt); % convert to a matlab sparse matrix
D = max(did);
W = max(wid);
N = sum(cnt);
% number of docs
% size of vocab
% total number of words
% It is often helpful to normalize by the document length:
Xn= X./repmat(sum(X,2),[1,W]) ; % divide word counts by doc length

%(a)
[z,c,sumd] = kmeans(Xn,20);

%(b)
%pick one best kmeans result
sumd2 = sumd;
for i = 1 : 10
    [z1,c1,sumd1] = kmeans(Xn,20);
    if sumd1 < sumd2
        z = z1;
        c = c1;
        sumd2 = sumd1;
    end
end

%(c)
%count the documents in each cluster
count = zeros(1,20);
for i = 1 : 20
   count(i) = sum(z==i); 
end

% get the most common words
for i = 1 : 20
[sorted,order] = sort( c(i,:), 2, 'descend');
fprintf('Cluster %d: ',i); fprintf('%s ',vocab{order(1:10)}); fprintf('\n');
end

%(d)
%find the clusters associated with document 1,15,30
m = [1,15,30];
[n,d] = size(z);
for i = 1: length(m)
   cluster = z(m(i));
   idx = 1;
   fprintf('%s','cluster');
   fprintf('%d\n',cluster);
   for j = 1: n
      if cluster == z(j)
          if idx> 12
              break
          end
          fprintf('%d\n',idx);
          fname = sprintf('data/text/example1/20000101.%04d.txt',j);
          txt = textread(fname,'%s',10,'whitespace','\r\n'); 
          fprintf('%s\n',txt{:});
          idx = idx + 1;
      end
   end
end
