function [ train_err, test_err ] = AdaBoost( X_tr, y_tr, X_te, y_te, n_trees )
%AdaBoost: Implement AdaBoost using decision stumps learned
%   using information gain as the weak learners.
%   X_tr: Training set
%   y_tr: Training set labels
%   X_te: Testing set
%   y_te: Testing set labels
%   n_trees: The number of trees to use

[n, ~] = size(X_tr);
[n_te,~] = size(X_te);

%make Y -1/+1
mu = mean(unique(y_tr));
y_tr = y_tr - mu;
y_te = y_te - mu;

%inital weight
w = ones(n,1);
w = w/n;
alphas = zeros(n_trees,1);
pre_tr = zeros(n,n_trees);
pre_te = zeros(n_te,n_trees);
result_tr = zeros(n,n_trees);
result_te = zeros(n_te,n_trees);

% iterations
for b = 1: n_trees
    % learn a weak hypothesis
    trees = fitctree(X_tr, y_tr,'Weights',w,'MaxNumSplits',1,'SplitCriterion','deviance');
    % precition
    pre_tr(:,b) = trees.predict(X_tr);
    pre_te(:,b) = trees.predict(X_te);
    % weighted misclassification error
    I = w'*(pre_tr(:,b)~=y_tr);
    % compute alphas
    alphas(b) = log((1-I)/I)/2;
    % update the weights
    zt = 2*sqrt(I*(1-I));
    w = w/zt.*exp(-alphas(b)*pre_tr(:,b).*y_tr);
    % predicate the final hypothesis
    result_tr(:,b) = sign(pre_tr*alphas);
    result_te(:,b) = sign(pre_te*alphas); 
end

y_trm = repmat(y_tr,1,n_trees);
E_tr = mean(result_tr~=y_trm);
y_tem = repmat(y_te,1,n_trees);
E_te = mean(result_te~=y_tem);

figure
x = 1:1:n_trees;
plot(x,E_tr,'-b',x,E_te,'r-');
title('Adaboost Classifaction Error');
legend('Training set error','Test set error');
xlabel('Number of Hypotheses');
ylabel('Error');

train_err = E_tr(n_trees);
test_err = E_te(n_trees);

end

