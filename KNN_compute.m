%k近邻
function [mknn_idx,mknn_nv]=KNN_compute(data,K)
[n,~]=size(data);
DM=squareform(pdist(data,"euclidean",2));
[dis_nv, dis_idx] = sort(DM, 2);
mknn_idx=dis_idx(:,2:K+1);
mknn_nv=dis_nv(:,2:K+1);