% Script to load data from zip.train, filter it into datasets with only one
% and three or three and five, and compare the performance of plain
% decision trees (cross-validated) and bagged ensembles (OOB error)
load zip.train;
te = load('zip.test');

fprintf('Working on the one-vs-three problem...\n\n');
subsample = zip(find(zip(:,1)==1 | zip(:,1) == 3),:);
Y = subsample(:,1);
X = subsample(:,2:257);
ct = fitctree(X,Y,'CrossVal','on');
fprintf('The cross-validation error of decision trees is %.4f\n', ct.kfoldLoss);
bee = BaggedTrees(X, Y, 200);
fprintf('The OOB error of 200 bagged decision trees is %.4f\n', bee);

% question c (1 vs 3)
% load test data
subsample_te = te(find(te(:,1)==1 | te(:,1) == 3),:);
Y_te = subsample_te(:,1);
X_te = subsample_te(:,2:257);
% run bagging with test
[ ~,  testErr] = BaggedTreesWithTest( X, Y, 200, X_te, Y_te );
fprintf('one-vs-three case, The test error of 200 bagged decision trees is %.4f\n', testErr);
% run single dicision tree
st = fitctree(X,Y);
testErr_st = mean(st.predict(X_te)~=Y_te);
fprintf('one-vs-three case, The test error of single decision tree is %.4f\n', testErr_st);

fprintf('\nNow working on the three-vs-five problem...\n\n');
subsample = zip(find(zip(:,1)==3 | zip(:,1) == 5),:);
Y = subsample(:,1);
X = subsample(:,2:257);
ct = fitctree(X,Y,'CrossVal','on');
fprintf('The cross-validation error of decision trees is %.4f\n', ct.kfoldLoss);
bee = BaggedTrees(X, Y, 200);
fprintf('The OOB error of 200 bagged decision trees is %.4f\n', bee);

% question c (3 vs 5)
% load test data
subsample_te = te(find(te(:,1)==3 | te(:,1) == 5),:);
Y_te = subsample_te(:,1);
X_te = subsample_te(:,2:257);
% run bagging with test
[ ~,  testErr] = BaggedTreesWithTest( X, Y, 200, X_te, Y_te );
fprintf('three-vs-five case, The test error of 200 bagged decision trees is %.4f\n', testErr);
% run single dicision tree
st = fitctree(X,Y);
testErr_st = mean(st.predict(X_te)~=Y_te);
fprintf('three-vs-five case, The test error of single decision tree is %.4f\n', testErr_st);
