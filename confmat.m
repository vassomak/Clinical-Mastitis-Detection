function [Accuracy, Precision, Recall, F1] = confmat(Ytest, label, modelname)

CF = confusionmat(Ytest,label, 'Order', [1,0]);
tp = CF(1,1);
tn = CF(2,2);
fp = CF(2,1);
fn = CF(1,2);
[Accuracy,Precision,Recall,F1] = metrics(tp, tn, fp, fn);
disp('Confusion Matrix: ')
disp(CF)
fprintf('Accuracy = %f\n', Accuracy);
fprintf('Precision = %f\n', Precision);
fprintf('Recall = %f\n', Recall);
fprintf('F1 = %f\n', F1);
figure;
classLabels = {'Mastitis', 'Healthy'};
cm = confusionchart(CF,classLabels);
str = sprintf('Confusion Matrix for %s',modelname);
cm.Title = str;
sortClasses(cm,{'Mastitis','Healthy'})
end