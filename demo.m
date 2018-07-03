% This is an example program for the paper: 
% 
% Lu Sun et al. Multi-label classification with meta-label specific features. ICPR-16. 
%
% The program shows how the MLSF program (The main function is "MLSF.m") can be used.
%
% The program was developed based on the following package:
%
% LIBSVM -- A Library for Support Vector Machines
% URL: http://www.csie.ntu.edu.tw/~cjlin/libsvm

%% Make experiments repeatedly
rng(1);

%% Add necessary pathes
addpath('data','eval');
addpath(genpath('func'));

%% Choose a dataset
dataset  =   'yeast';
load([dataset,'.mat']);

%% Set parameters 
opts.size     = 10;               
opts.epsilon  = 1e-2; 
opts.alpha    = 0.8;
opts.gamma    = 1e-2;
opts.rho      = 1;

hm = [];
rl = [];
cv = [];
oe = [];
ap = [];

%% Perform n-fold cross validation
num_fold = 10; Results = zeros(5,num_fold);
indices = crossvalind('Kfold',size(data,1),num_fold);
for i = 1:num_fold
    disp(['Fold ',num2str(i)]);
    test = (indices == i); train = ~test; 
    [Outputs,Pre_Labels] = MLSF(data(train,:),target(:,train),data(test,:),opts);
    hm=[hm,Hamming_loss(Pre_Labels,target(:,test))];
    rl=[rl,Ranking_loss(Outputs,target(:,test))];
    oe=[oe,One_error(Outputs,target(:,test))];
    cv=[cv,coverage(Outputs,target(:,test))];
    ap=[ap,Average_precision(Outputs,target(:,test))];
end

HammingLoss=mean(hm);
RankingLoss=mean(rl);
OneError=mean(oe);
Coverage=mean(cv);
Average_Precision=mean(ap);
