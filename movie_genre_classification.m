clc; clear; close all;

%% Load dataset
data = readtable('movie_dataset.csv','TextType','string');

%% Show detected columns (optional)
vars = data.Properties.VariableNames;
disp('Detected columns:');
disp(vars.');

%% Prepare: convert all non-numeric columns (except Title) to numeric
for i = 1:length(vars)
    v = vars{i};
    if strcmpi(v,'Title')
        continue; % keep title (we'll remove later)
    end

    col = data.(v);

    % If already numeric, skip
    if isnumeric(col)
        continue;
    end

    % If string array or cellstr
    if isstring(col) || iscellstr(col) || iscell(col)
        % Convert cell -> string for easier processing
        colStr = string(col);

        % Replace missing/empty with a placeholder
        colStr(colStr=="" | ismissing(colStr)) = "<missing>";

        % Check if all entries look numeric (integers/floats)
        isNumLike = all(~ismissing(str2double(strrep(colStr,',',''))));

        if isNumLike
            % Convert numeric-like strings to numbers
            numVals = str2double(strrep(colStr,',',''));
            data.(v) = numVals;
        else
            % Non-numeric: encode as categorical indices
            c = categorical(colStr);
            data.(v) = double(c); % numeric codes for categories
        end
    elseif iscategorical(col)
        data.(v) = double(col);
    else
        % fallback: try to convert via double
        try
            data.(v) = double(col);
        catch
            % last resort: convert to categorical then numeric
            data.(v) = double(categorical(string(col)));
        end
    end
end

%% Remove Title if present
if ismember('Title', data.Properties.VariableNames)
    data.Title = [];
end

%% Ensure Genre is the last column and is numeric codes
% (If Genre exists as non-numeric, ensure it's categorical-coded)
if ismember('Genre', data.Properties.VariableNames)
    if ~isnumeric(data.Genre)
        data.Genre = double(categorical(string(data.Genre)));
    end
    % move Genre to last if it's not already
    vars = data.Properties.VariableNames;
    if ~strcmp(vars{end}, 'Genre')
        % reorder so Genre is last
        other = setdiff(vars, 'Genre', 'stable');
        data = data(:, [other 'Genre']);
    end
end

%% Now convert table to array
X = table2array(data(:, 1:end-1));
y = table2array(data(:, end));

%% Handle any NaNs in X (simple strategy)
if any(isnan(X),'all')
    % replace NaN by column median (numeric columns)
    for colIdx = 1:size(X,2)
        col = X(:,colIdx);
        if any(isnan(col))
            med = nanmedian(col);
            if isnan(med)
                med = 0;
            end
            col(isnan(col)) = med;
            X(:,colIdx) = col;
        end
    end
end

%% Train-test split
cv = cvpartition(size(X,1),'HoldOut',0.2);
idx = cv.test;

X_train = X(~idx,:);
X_test  = X(idx,:);
y_train = y(~idx,:);
y_test  = y(idx,:);

%% Train Random Forest (TreeBagger)
model = TreeBagger(200, X_train, y_train, 'Method','classification', 'OOBPrediction','On');

%% Predict
y_pred = str2double(predict(model, X_test));

%% Accuracy
accuracy = sum(y_pred == y_test) / numel(y_test);
fprintf('Model Accuracy: %.2f%%\n', accuracy*100);

%% Confusion matrix
figure;
confusionchart(y_test, y_pred);
title('Confusion Matrix - Movie Genre Classification');

%% Feature importance
imp = model.OOBPermutedPredictorDeltaError;
figure;
bar(imp);
title('Feature Importance');
xlabel('Feature Index');
ylabel('Importance Score');

%% Show feature names (for reference)
disp('Feature order used (columns):');
disp(data.Properties.VariableNames(1:end-1).');
