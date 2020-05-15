function [ oobErr ] = BaggedTrees( X, Y, numBags )
%BAGGEDTREES Returns out-of-bag classification error of an ensemble of
%numBags CART decision trees on the input dataset, and also plots the error
%as a function of the number of bags from 1 to numBags
%   Inputs:
%       X : Matrix of training data
%       Y : Vector of classes of the training examples
%       numBags : Number of trees to learn in the ensemble
%
%   You may use "fitctree" but do not use "TreeBagger" or any other inbuilt
%   bagging function


% make Y +1/-1
mu = mean(unique(Y));
Y = Y - mu;


n = size(X,1); % number of observations
votes = zeros (n,numBags);
draw = randi(n,n,numBags); % draw matrix

for b = 1:numBags
    % Sampling data
    dataX = X(draw(:,b),:);
    dataY = Y(draw(:,b),:);
    %learn tree
    tree = fitctree(dataX,dataY);
    % store the prediction data
    H = tree.predict(X);
    % a multiplier that will mult 1 to those predictions for which observation 
    % was not in the bootstrap sample
    myMult = zeros(n,1); 
    for i = 1:n 
        if sum(draw(:,b) == i)==0 % if I cannot find any x_i draw 
            myMult(i) = 1;
        end
    end 
    % put things togther
    votes(:,b) = myMult.*H;
end

%cumulative summation
for b = 2:numBags
    votes(:,b) = votes(:,b) + votes(:,b-1);
end

Result = sign(votes);
yExtend = repmat(Y,1,numBags);
toPlot = mean(Result~=yExtend);

figure
x = 1:1:numBags;
plot(x,toPlot,'-b');
title('Out Of Bag Error');
xlabel('Number of bags');
ylabel('OOBerror');


oobErr = mean(Result(:,numBags)~=Y);
end
