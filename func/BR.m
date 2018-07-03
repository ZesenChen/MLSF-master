function Pre_Labels = BR(train_data,train_target,test_data,test_target)
%BR Binary Relevance [1] with LIBSVM [2]
%       Input:
%           train_data       An N x D data matrix, each row denotes a sample
%           train_target     An L x N label matrix, each column is a label set
%           test_data        An Nt x D test data matrix, each row is a test sample
%       Output:
%           Pre_Labels       An L x Nt predicted label matrix, each column is a predicted label set
%
%  [1] M.R. Boutel et al. Learning multi-label scene classification. Pattern Recognition, 2004.
%
%  [2] C. Chang and C. Lin. LIBSVM : a library for support vector machines. ACM Transactions on Intelligent Systems and Technology, 2011

[num_label,num_test] = size(test_target);
Pre_Labels = zeros(num_label,num_test);    

for i=1:num_label
    model = svmtrain(train_target(i,:)',train_data,'-t 0 -q');
    Pre_Labels(i,:) = svmpredict(test_target(i,:)',test_data,model,'-q')';
end

end

