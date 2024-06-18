function [RNN] =RNN_compute(data,K)
 %% 反近邻(Reverse Neighbors)
[n,~]=size(data);
DM=squareform(pdist(data,"euclidean",2));
[~, dis_idx] = sort(DM, 2);
mknn_idx=dis_idx(:,2:K+1);
RNN=cell(n,1);
for i=1:n
    rnnnum=mknn_idx(i,:);
    for j=1:K
        if i~=j
            if any(mknn_idx(rnnnum(1,j),:)==i)
                RNN{i}=[RNN{i},rnnnum(1,j)];
            end
        end
    end
end