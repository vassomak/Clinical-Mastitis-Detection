 clc; clear; tic;

mastitis = readtable('mastitis.csv', 'ReadVariableNames', true, 'PreserveVariableNames', true);
flag = 0;
[xdata, ydata] = preprocessing(mastitis, flag);

Xdata = table2array(xdata);
Ydata = table2array(ydata);

%% Train - Test split 

p = 0.3;
[train,test] = crossvalind('HoldOut', Ydata, p);
Xtrain = Xdata(train,:);
Ytrain = Ydata(train,1);
Xtest = Xdata(test,:);
Ytest = Ydata(test,1);


%% ----- SVM -----
% Grid search and cross-validation for parameters tuning

numFolds = 5;
Ind = crossvalind('Kfold', Ytrain, numFolds);

C = 2.^(-5:1:5);
gamma = 2.^(-3:1:5);

accFoldSVM = zeros(length(C), length(gamma));

      for j = 1 : length(C)
        for k = 1 : length(gamma)       
                 for i = 1:numFolds

            XtestFold = Xtrain(Ind==i,:);
            YtestFold = Ytrain(Ind==i,:);
    
            XtrainFold = Xtrain(Ind~=i,:);
            YtrainFold = Ytrain(Ind~=i,:);
    
             svm = fitcsvm(XtrainFold,YtrainFold, 'KernelFunction', 'rbf', 'KernelScale', gamma(k), ...
             'BoxConstraint', C(j));

             Ypred_svm(Ind==i,1) = predict(svm, XtestFold);

                 end
         accFoldSVM(j,k) = sum(grp2idx(Ypred_svm) == grp2idx(Ytrain)) / length(Ytrain); 

        end
      end
      

%% Determine best C and gamma values for SVM

    [maxCol, ICol] = max(accFoldSVM, [], 1);
    [~, IRow] = max(maxCol, [] ,2);
    bestGamma= gamma(IRow);
    bestC = C(ICol(IRow));
    
%% ----- KNN -----    
% Cross-validation for 'k' tuning
k = 1 : 2: 30;
accFoldKNN = zeros(length(k),1);

for j = 1 : length(k)
    for i = 1 : numFolds
        
            XtestFold = Xtrain(Ind==i,:);
            YtestFold = Ytrain(Ind==i,:);
    
            XtrainFold = Xtrain(Ind~=i,:);
            YtrainFold = Ytrain(Ind~=i,:);
            
            knn = fitcknn(XtrainFold, YtrainFold,'NumNeighbors', k(j));
            
            Ypred_knn(Ind==i,1) = predict(knn, XtestFold);
    end
             accFoldKNN(j) = sum(grp2idx(Ypred_knn) == grp2idx(Ytrain)) / length(Ytrain); 
end

  [~, idx] = max(accFoldKNN); 
  bestK = k(idx);
  
  
  %% ----- TREE -----
  
  % Cross Validation for MinParentSize tuning

MinParentSize = 1:25;
accFold = zeros(length(MinParentSize),1);

for j = 1 : length(MinParentSize)
    for i = 1 : numFolds
        
            XtestFold = Xtrain(Ind==i,:);
            YtestFold = Ytrain(Ind==i,:);
    
            XtrainFold = Xtrain(Ind~=i,:);
            YtrainFold = Ytrain(Ind~=i,:);
            
            tree = fitctree(XtrainFold, YtrainFold,'MinParentSize', MinParentSize(j));
            
            Ypred_tree(Ind==i,1) = predict(tree, XtestFold);
    end
             accFold(j) = sum(grp2idx(Ypred_tree) == grp2idx(Ytrain)) / length(Ytrain); 
end


%% Determine best minParentSize
[~, idx] = max(accFold); 
bestMinParentSize = MinParentSize(idx);

%% Cross Validation for MinLeafSize Tuning, given MinParentSize
MinLeafSize = 1:25;
accFold = zeros(length(MinLeafSize),1);

for j = 1 : length(MinLeafSize)
    for i = 1 : numFolds
        
            XtestFold = Xtrain(Ind==i,:);
            YtestFold = Ytrain(Ind==i,:);
    
            XtrainFold = Xtrain(Ind~=i,:);
            YtrainFold = Ytrain(Ind~=i,:);
            
            tree = fitctree(XtrainFold, YtrainFold,'MinParentSize', bestMinParentSize, 'MinLeafSize', MinLeafSize(j));
            
            Ypred_tree(Ind==i,1) = predict(tree, XtestFold);
    end
             accFold(j) = sum(grp2idx(Ypred_tree) == grp2idx(Ytrain)) / length(Ytrain); 
end

%% Determine best minleafSize
[~, idx] = max(accFold); 
bestMinLeafSize= MinLeafSize(idx);

  
%% Train the models

mdlSVM = fitcsvm(Xtrain, Ytrain, 'KernelFunction', 'rbf', 'KernelScale', bestGamma, ...
    'BoxConstraint', bestC);

mdlKNN = fitcknn(Xtrain, Ytrain,'NumNeighbors', bestK);
mdlTREE = fitctree(Xtrain, Ytrain, 'MinParentSize', bestMinParentSize, 'MinLeafSize',bestMinLeafSize);
mdlNB = fitcnb(Xtrain, Ytrain);

%% Test - Evaluate on the Testing Set

%% SVM
[label, ~] = predict(mdlSVM, Xtest);
mdlsvm = fitPosterior(mdlSVM);
disp('---- SVM ----')
[X_svm,Y_svm,AUC_svm] = roc(mdlsvm, Ytrain, 'SVM',0);
[Accuracy_svm, Precision_svm, Recall_svm, F1_svm] = confmat(Ytest, label, 'SVM');

%% KNN
[label, ~] = predict(mdlKNN, Xtest);
disp('---- KNN ----')
[X_knn,Y_knn,AUC_knn] = roc(mdlKNN, Ytrain, 'K-NN',0);
[Accuracy_knn, Precision_knn, Recall_knn, F1_knn] = confmat(Ytest, label, 'K-NN');

%% Decision Tree
[label, ~] = predict(mdlTREE, Xtest);
disp('---- Decision Tree ----')
[X_tree,Y_tree,AUC_tree] = roc(mdlTREE, Ytrain, 'Decision Tree',0);
[Accuracy_tree, Precision_tree, Recall_tree, F1_tree] = confmat(Ytest, label, 'Decision Tree');

%% Naive Bayes 
[label, ~] = predict(mdlNB, Xtest);
disp('---- Naive Bayes ----')
[X_nb,Y_nb,AUC_nb] = roc(mdlNB, Ytrain, 'Naive Bayes',0);
[Accuracy_nb, Precision_nb, Recall_nb, F1_nb] = confmat(Ytest, label, 'Naive Bayes');

%% --- PLOT ROC ---
figure
x = [0,0,1];
y = [0,1,1];
plot(X_svm,Y_svm,':', X_knn,Y_knn, X_tree,Y_tree,'--',X_nb,Y_nb,'--', 'LineWidth',1.5);
legend( 'SVM', 'K-NN', 'Decision Tree', 'Naive Bayes')
xlabel('False positive rate') 
ylabel('True positive rate')
ylim([-0.1 1.1])
xlim([-0.1 1.1])
title('ROC Curve')
toc;

