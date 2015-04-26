function [trndata,tstdata]=scaleSVM(trndata,tstdata,Lower,Upper)
%--------------------------------------------------------------------------
% DESCRIPTION: Used to Scale data uniformly
%--------------------------------------------------------------------------
Data=trndata.X;
[MaxV, I]=max(Data);
[MinV, I]=min(Data);
[R,C]= size(Data);
scaled=(Data-ones(R,1)*MinV).*(ones(R,1)*((Upper-Lower)*ones(1,C)./(MaxV-MinV)))+Lower;
for i=1:size(Data,2)
    if(all(isnan(scaled(:,i))))
        scaled(:,i)=0;
    end
end
trndata.X=scaled;

%###### SCALE THE TEST DATA TO THE RANGE OF TRAINING DATA ###########
Data=tstdata.X;
[R,C]= size(Data);
scaled=(Data-ones(R,1)*MinV).*(ones(R,1)*((Upper-Lower)*ones(1,C)./(MaxV-MinV)))+Lower;
for i=1:size(Data,2)
    if(all(isnan(scaled(:,i))))
        scaled(:,i)=0;
    end
end
tstdata.X=scaled;

% Data = trndata.Y;
% [MaxV, I]=max(Data);
% [MinV, I]=min(Data);
% [R,C]= size(Data);
% scaled=(Data-ones(R,1)*MinV).*(ones(R,1)*((Upper-Lower)*ones(1,C)./(MaxV-MinV)))+Lower;
% for i=1:size(Data,2)
%     if(all(isnan(scaled(:,i))))
%         scaled(:,i)=0;
%     end
% end
% trndata.Y=scaled;