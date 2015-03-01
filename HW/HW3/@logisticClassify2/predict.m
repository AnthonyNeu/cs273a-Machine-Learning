function Yte = predict(obj,Xte)
% Yhat = predict(obj, X)  : make predictions on test data X

% (1) make predictions based on the sign of wts(1) + wts(2)*x(:,1) + ...
% (2) convert predictions to saved classes: Yte = obj.classes( [1 or 2] );

[n,d] = size(Xte);
Xte = [ones(n,1), Xte];      % extend features X by the constant feature
value = Xte * obj.wts';
Yte = zeros(size(value,1),1);

for i = 1: size(value,1)
   if value(i) >0
       Yte(i) = obj.classes(2);
   else
       Yte(i) = obj.classes(1);
   end
end
