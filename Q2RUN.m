load zip.train;
te = load('zip.test');

fprintf('Working on the one-vs-three problem...\n\n');
subsample_tr = zip(find(zip(:,1)==1 | zip(:,1) == 3),:);
Y_tr = subsample_tr(:,1);
X_tr = subsample_tr(:,2:257);
subsample_te = te(find(te(:,1)==1 | te(:,1) == 3),:);
Y_te = subsample_te(:,1);
X_te = subsample_te(:,2:257);
[ train_err, test_err ] = AdaBoost( X_tr, Y_tr, X_te, Y_te, 200 );

fprintf('Working on the three-vs-five problem...\n\n');
subsample_tr = zip(find(zip(:,1)==3 | zip(:,1) == 5),:);
Y_tr = subsample_tr(:,1);
X_tr = subsample_tr(:,2:257);
subsample_te = te(find(te(:,1)==3 | te(:,1) == 5),:);
Y_te = subsample_te(:,1);
X_te = subsample_te(:,2:257);
[ train_err, test_err ] = AdaBoost( X_tr, Y_tr, X_te, Y_te, 200 );