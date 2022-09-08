
 
function [xdata, ydata] = preprocessing(mastitis, flag)

% flag: if flag is 1, then the correlation plot is shown

%% Delete Cow_ID, Day, Breed, Hardness, Pain, Milk_Visibility and duplicates - balance the dataset
x = [6:14, 18];
mastitis = unique(mastitis(:, x), 'rows', 'stable');

%onesLabel = mastitis(mastitis.class1 == 1, :);
%zerosLabel = mastitis(mastitis.class1 == 0, :);

%b = size(zerosLabel,1);
%z = size(onesLabel,1);

%data = [onesLabel; zerosLabel(randperm(b, 3*z),:)];
%newdata = data(randperm(size(data, 1)), :);
%% Scale to [0 1] 
xdata = normalize(mastitis(:,1:end-1), 'range');
ydata = mastitis(:,end);

%% Feature selection
if flag == 1
    figure
%corrplot([xdata ydata])
r = corrcoef(table2array([xdata ydata]));
labels = [xdata.Properties.VariableNames 'class'];
xvalues = labels;
yvalues = labels;
h = heatmap(xvalues,yvalues,r);
h.Title = 'Correlation Matrix';
end

col = [2,4,6,9] ;
xdata = xdata(:, col);
final = unique([xdata ydata]);
xdata = final(:,1:end-1);
ydata = final(:,end);
% Cow_id, Day and Breed are dropped
% Months after giving birth and Previous mastitis status: very low
% correlation with the class, so they are dropped

% IUFL and EUFL: same correlation with the class (0.46) and highly correlated to each
% other (0.99), so EUFL is dropped

% IUFR and EUFR: same correlation with the class (0.45) and highly correlated to each
% other (0.99), so EUFR is dropped

% IURL and EURL: same correlation with the class (0.19), and highly correlated to each
% other (0.98), so EURL is dropped

% IURR and EURR: very low correlation with the class (0.07), so they are both dropped

% Temperature: high correlation with the class (0.7), so it's kept (?)

% Hardness, Pain and Milk visivility: all too highly correlated with the
% class (>0.96) and with each other, so all dropped

end