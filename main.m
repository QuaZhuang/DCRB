 clc;
 clear;
%load('jain.mat')
load('compound.mat')
%K=21;
K=16;
%data=normalization(data);
[cluster,~]=DCRB( data,K);
gscatter(data(:, 1),data(:, 2),cluster);