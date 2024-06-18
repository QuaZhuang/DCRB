function [MNN] =MNN(data,K)
[n,~]=size(data);
DM=squareform(pdist(data,"euclidean",2));
[~, dis_idx] = sort(DM, 2);
mknn_idx=dis_idx(:,2:K+1);
%% 互近邻(mutual nearest neighbor)
[n,~]=size(data);
MNN=cell(n,1);
%各个点的互近邻个数不一，故用元胞，每一个元胞存储一个点的互近邻
for i=1:n
    mnnnum=mknn_idx(i,:);
    for j=1:K
        if any((mknn_idx(mnnnum(1,j),:))==i)
            MNN{i}=[MNN{i},mnnnum(1,j)];
        end
    end
end


