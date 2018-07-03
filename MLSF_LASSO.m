function V = MLSF_LASSO( X,Y,K,m,gamma,rho )
%MLSF_LASSO Specific-features mining via Lasso with ADMM [1]
%
%    Syntax
%
%       V = MLSF_LASSO(X,Y,K,m,gamma,rho)
%
%    Description
%
%       Input:
%           X       An N x D data matrix, each row denotes a sample
%           Y       An L x N label matrix, each column is a label set
%           K       The number of meta-labels
%           m       The meta-label membership
%           gamma   The sparsity parameter of Lasso
%           rho     An parameter for ADMM
% 
%       Output:
%           V       An D x L Sparse regression parameter matrix
%
%  [1] S. Boyd et al. Distributed optimization and statistical learning via 
%      the alternating direction method of multipliers. 
%      Foundations and Trends in Machine Learning, 2011.

%% Get the size of data
[num_data, num_feature] = size(X);

%% Transform Y into meta-label matrixZ
Z = zeros(K, num_data);
for i = 1:K
    meta_Z = Y((m==i),:);
    meta_size = size(meta_Z,1);
    if meta_size > 1
        [~, ~, index] = unique(bi2de(meta_Z'));
        index = index - 1;
        Z(i,:) = index./ max(index);
    else
        Z(i,:) = meta_Z;
    end
end

%% Solve Lasso by ADMM
% Global constants
MAX_ITER = 100;
ABSTOL   = 1e-4;
RELTOL   = 1e-2;

% Cache some results
if num_data >= num_feature
    L = chol((X'*X + rho*speye(num_feature)),'lower');
    isSkinny = 1;  
else
    L = chol((speye(num_data) + 1/rho*(X*X')),'lower');
    isSkinny = 0;  
end
L = sparse(L); U = sparse(L');
XtZ = (Z*X)';
sqABSTOL = sqrt(num_feature)*ABSTOL;
kappa = gamma / rho;

% ADMM solver
V = zeros(num_feature,K); 
G = zeros(num_feature,K);
for k = 1:MAX_ITER
    
    % W-update
    q = XtZ + rho*(V - G);
    if isSkinny
        W = U \ (L \ q);
    else
        W = q/rho - (X'*(U \ ( L \ (X*q) )))/rho^2;
    end
    
    % V-update with relaxation
    Vold = V; Wold = W;
    W = Wold + G;
    V = max(0,W-kappa) - max(0,-W-kappa);
    
    % G-update
    G = G + (Wold - V);
    
    % Check the conditions of termination
    r_norm  = norm(W - V);
    s_norm  = norm(-rho*(V - Vold));
    eps_pri = sqABSTOL + RELTOL*max(norm(W), norm(-V));
    eps_dual= sqABSTOL + RELTOL*norm(rho*G);
    if (r_norm < eps_pri && s_norm < eps_dual)
        break;
    end
    
end

end
