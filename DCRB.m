function [Cluster,time]=DCRB(data,K)
tic;
[n,~]=size(data);
[~,mknn_nv]=KNN_compute(data,K);
K_ridus=mknn_nv(:,K);
MNN=MNN_compute(data,K);
RNN=RNN_compute(data,K);
DM=squareform(pdist(data,"euclidean",2));
R_ridus=zeros(n,1);
for i=1:n
    if isempty(RNN{i})
        R_ridus(i)=0;
    else
        R_ridus(i)=max(DM(i,RNN{i}));
    end
end
weight_DC=zeros(n,n);
for i=1:n
    weight_DC(i,RNN{i})=R_ridus(RNN{i})./R_ridus(i);
end
LX=cell(n,1);
for i=1:n
    LX{i}=find(weight_DC(i,:)>0);
end
LX_num=sum(weight_DC~= 0);
weight_avDC=zeros(n,1);
for i=1:n
    if LX_num(i)==0
        weight_avDC(i)=0;
    else
        weight_avDC(i)=sum(weight_DC(i,:),2)./LX_num(i);
    end
end
RAll=zeros(n,1);
for i=1:n
       RAll(i)=sum(DM(i,RNN{i}));
end
p=R_ridus./RAll;
h=zeros(n,1);
for i=1:n
    h(i)=-log2(p(i));
end
ah=sum(h)./n;
[idx,~]=find(h>ah);

Object_C=find(weight_avDC>1);

CC=unique(union(Object_C,idx));
FC=intersect(Object_C,idx);
OB=setdiff(CC,FC);
Object_C=FC;
oc=zeros(n,1);
oc(Object_C)=1;

 [~,idx2]=sort(h(Object_C),'descend');
Object_C=Object_C(idx2);
m1 = Object_C;
Cluster = zeros(n,1);
p = 1;
while ~isempty(Object_C)
   
    c=intersect(Object_C,LX{Object_C(1)});
    c=[c;Object_C(1)];
    Cluster(c) = p;
   
    gg = setdiffplus(c,Object_C(1));
    Object_C = setdiffplus(Object_C,c);
    if isempty(Object_C)
        break
    end

    while 1
        if isempty(gg)
            break
        end
        hh = find( weight_DC(gg(1),:)>0);
        ff = intersect(Object_C,hh);
        if isempty(ff)
            gg(1) = [];
            continue
        end
        Cluster(ff) = p;

        Object_C = setdiffplus(Object_C,ff);
        gg = [gg;ff];
        gg(1) =[];
    end
    p = p + 1;
end

[~,idx3]=sort(h(OB),'descend');
OB=OB(idx3);
p=p+1;
while ~isempty(OB)
    aa=OB(1);
    bb=LX{aa};
    idx4=find(Cluster(bb)>0);
    cc=bb(idx4);
    if isempty(cc)
    c=intersect(OB,LX{OB(1)});
    c=[c;OB(1)];
    Cluster(c)=p;
  
    gg = setdiffplus(c,OB(1));
    OB = setdiffplus(OB,c);
    if isempty(OB)
        break
    end
   
    while 1
        if isempty(gg)
            break
        end
        hh = find( weight_DC(gg(1),:)>0);
        ff = intersect(OB,hh);
        if isempty(ff)
            gg(1) = [];
            continue
        end
        Cluster(ff) = p;
        %gscatter(data(:,1),data(:,2),Cluster);
        OB = setdiffplus(OB,ff);
        gg = [gg;ff];
        gg(1) =[];
    end
    else
        [~,dd]=min(DM(aa,cc));
        Cluster(aa)=Cluster(cc(dd));
        %gscatter(data(:,1),data(:,2),Cluster,color);
        %legend('off');
        OB(1)=[];
    end
    p=p+1;
end
%gscatter(data(:,1),data(:,2),Cluster,color);
kk=unique(Cluster);
kk_num=numel(kk);
for ww=8:kk_num
    bbbb=kk(ww,1);
    aaaa=numel(find(Cluster(:,1)==bbbb));
    if aaaa<K/3
        [idx8,~]=find(Cluster==bbbb);
        Cluster(idx8)=0;
    end
end
%gscatter(data(:,1),data(:,2),Cluster,color);

RM_point=find(Cluster==0);
[~,idx5]=sort(h(RM_point),'ascend');
RM_point=RM_point(idx5);
nnn=numel(RM_point);
for i=1:nnn
    aaa=RM_point(i);
    oo=RNN{aaa};
    ii=find(Cluster(oo)>0);
    ooo=oo(ii);
    if isempty(ooo)
        qq=find(Cluster>0);
        [~,idx7]=min(DM(aaa,qq));
        Cluster(aaa)=Cluster(qq(idx7));
        %gscatter(data(:,1),data(:,2),Cluster,color);
    else
        jj=unique(Cluster(ooo));
        n4=numel(jj);
        ppp=[];
        for j=1:n4
            w=numel(find(Cluster(ooo)==jj(j)));
            ppp=[ppp;w];
        end
        [~,idx7]=max(ppp);
        Cluster(aaa)=jj(idx7); 
        %gscatter(data(:,1),data(:,2),Cluster,color);
    end
end
time=toc;
% gscatter(data(:,1),data(:,2),Cluster,color);
% legend('off');
% allidex=all_idex(label,Cluster);