function plot2DLinear(obj, X, Y)
% plot2DLinear(obj, X,Y)
%   plot a linear classifier (data and decision boundary) when features X are 2-dim
%   wts are 1x3,  wts(1)+wts(2)*X(1)+wts(3)*X(2)
%
  [n,d] = size(X);
  if (d~=2) error('Sorry -- plot2DLogistic only works on 2D data...'); end;

  %%% TODO: Fill in the rest of this function...  
  figure;
  scatter(X(:,1),X(:,2),50,Y,'filled');
  as = axis;
  
 
  xs = as(1):0.05:as(2)';
  hold on;
  plot(xs,- obj.wts(1)/obj.wts(3) + xs * - obj.wts(2)/obj.wts(3),'g-');
  axis(as);
  
  
  
  