function [X,Y,AUC] = roc(model, ytrain, modelname,flag)

[~,score] = resubPredict(model);
[X,Y,~,AUC,~] = perfcurve(ytrain,score(:,2),1);
fprintf('AUC = %f\n', AUC);

if flag==1
x_p = [0 0 1];
y_p = [0 1 1];
plot(X,Y, x_p, y_p, 'LineWidth', 1.5)
ylim([-0.1 1.1])
xlim([-0.1 1.1])
xlabel('False positive rate') 
ylabel('True positive rate')
str = sprintf('ROC Curve for Classification by %s',modelname);
title(str)
legend(sprintf(modelname), 'Perfect Classifier')
end
end