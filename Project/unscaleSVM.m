function [Y]=unscaleSVM(Y,Ytr,MinV,MaxV)
%--------------------------------------------------------------------------
% DESCRIPTION: Used to do the revert of scaling
%--------------------------------------------------------------------------

Data = Y;
[Lower, I]=max(Ytr);
[Upper, I]=min(Ytr);
[R,C]= size(Data);
unscaled=(Data-ones(R,1)*MinV).*(ones(R,1)*((Upper-Lower)*ones(1,C)./(MaxV-MinV)))+Lower;
for i=1:size(Data,2)
    if(all(isnan(unscaled(:,i))))
        unscaled(:,i)=0;
    end
end
Y=unscaled;