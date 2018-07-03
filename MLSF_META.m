function m = MLSF_META( X,Y,alpha,epsilon,K )
%MLSF_META Meta-label learning for MLSF by spectral clustering [1]
%
%    Syntax
%
%       m = MLSF_META( X,Y,alpha,epsilon,K )
%
%    Description
%
%       Input:
%           X        An N x D data matrix, each row denotes a sample
%           Y        An L x N label matrix, each column is a label set
%           alpha    Importance factor for computing affinity matrix
%           epsilon  Threshold of epsilon-neighborhood
%           K        Number of meta-labels
% 
%       Output:
%           m       An L x 1 meta-label membership vector
%
%  [1] M. Belkin and P. Niyogi. Laplacian eigenmaps and spectral techniques 
%      for embedding and clustering. NIPS, 2001.

%% Construct affinity matrix
% Label similarity
A1 = 1 - pdist(Y,'jaccard');
% Instance locality
label_mean = bsxfun(@rdivide,Y*X,sum(Y,2));
A2 = exp(-pdist(label_mean));
% Label similarity comes from two folds
A = alpha.*A1 + (1-alpha).*A2;
% epsilon-eighborhoods
A(A<epsilon) = 0; A(isnan(A)) = 0;
% Affinity matrix
A = sparse(squareform(A));

%% Apply spectral clustering
% Compute degree matrix
num_label = size(A,1);
D = sum(A,2); D(D==0) = eps;
D = spdiags(1./sqrt(D),0,num_label,num_label);
% Compute Laplacian matrix(for largest eigenvectors)
L = D * A * D;
% Compute the eigenvectors
[U, ~] = eigs(L,K,'LM');
% Normalize the eigenvectors row-wise
U = bsxfun(@rdivide, U, sqrt(sum(U.^2, 2)));
% Apply kmeans on U in row-wise
m = kmeans(U,K,'MaxIter',20,'OnlinePhase','off');

end

