%%% example of applying bft-based method on miRNA for outcome prediction task
clearvars; clc;

%% load libraries and folders
addpath('/shared/radon/TOP/mrmr_bft_miRNAs/FEAST/matlab');
addpath('/shared/radon/TOP/mrmr_bft_miRNAs/dataset');
addpath('/shared/radon/TOP/mrmr_bft_miRNAs/MIToolbox/matlab');

%% load datasets
%feats = load('/home/zongfan2/Documents/clips/Volumes/Seagate/seizure_detection/competition_data/bft_data/train_feature_2.txt'); % feature dataset
%labels = load('/home/zongfan2/Documents/clips/Volumes/Seagate/seizure_detection/competition_data/bft_data/train_label_2.txt');  % outcome labels: alive 0 (negative), dead 1 (positive)
%miRNA_Names = strsplit(strtrim(fileread('/home/zongfan2/Documents/clips/Volumes/Seagate/seizure_detection/competition_data/bft_data/feat_2.txt')));

% feats = load('/shared/radon/TOP/mrmr_bft_miRNAs/dataset/miRNA_features.txt'); % feature dataset
% labels = load('/shared/radon/TOP/mrmr_bft_miRNAs/dataset/death_labels.txt');  % outcome labels: alive 0 (negative), dead 1 (positive)
% miRNA_Names = strsplit(strtrim(fileread('/shared/radon/TOP/mrmr_bft_miRNAs/dataset/miRNA_names.txt')));

datasetPath = '/shared/anastasio-s2/SI/TCVAE/DL_feature_interpretation/result/batch1_to_10/all_features.csv';
opts = detectImportOptions(datasetPath);
data = readtable(datasetPath, opts);

% Separate features (X) and labels
X = table2array(data(:, 1:end-1));
imageNames = data{:, end};
y = zeros(size(imageNames)); % Initialize labels array

% Extract labels from image names
for i = 1:length(imageNames)
    if contains(imageNames{i}, '-istumor1')
        y(i) = 1; % Tumor
    elseif contains(imageNames{i}, '-istumor0')
        y(i) = 0; % Not tumor
    else
        error('Unknown label for image %s', imageNames{i});
    end
end

% Split data into training and testing sets
cv = cvpartition(size(X, 1), 'HoldOut', 0.3);
idxTrain = training(cv);
idxTest = test(cv);
Xtrn = X(idxTrain, :);
Ytrn = y(idxTrain, :);
Xtst = X(idxTest, :);
Ytst = y(idxTest, :);

%% filter-based selection + BFT-based selection
% Criteria = {'mim', 'jmi','disr', ...
%                     'condred', 'mrmr', 'cmim', ...
%                     'mifs', 'icap', 'cife', 'relief'};  % the set of filter-based methods to be considered
Criteria = {'mrmr'};
Perfs = {};
Num_Pre_Feats = 13;
for Cri_Indx = 1:length(Criteria)
	% Pre-selection with a filter-based selection method
	fprintf(['>>>>> Evaluate ', Criteria{Cri_Indx}, '+BFT >>>>>\n'])
	X_Round = round(Xtrn);  % discretize the feature values
	Pre_Feat_Indices = feast(Criteria{Cri_Indx},Num_Pre_Feats,X_Round,Ytrn);

	delta = 1;  lambda = 0.07;  K = 5 ;  B = 1;  % Parameters in BFT
	Pre_Indx = [];

	% Re-fine selected subset with the BFT-based selection method
    % delta: control the class balancing; generate (major - minor) * delta in the training set; delta = 1 represent to generate samples so that the major and minor class have the same samples
    % lambda: penalty to determine the sparsity
    % K: number of neighbors
    % B: times of resampling and bft running
	delta = 1;  lambda = 0.07;  K = 5 ;  B = 1;  % Parameters in BFT
	Pre_Indx = [];  % specific features that must be selected in the final set; set to empty
	% Run BFT to further select better features
	Xtrn_BFT = Xtrn(:, Pre_Feat_Indices); Ytrn_BFT = Ytrn+1;  % label: 1 (negative) and 2 (positive) for BFT
	Xtst_BFT = Xtst(:, Pre_Feat_Indices); Ytst_BFT = Ytst+1;  % label: 1 (negative) and 2 (positive) for BFT
	[~, idx_fs,~,~] = bft(Xtrn_BFT,Ytrn_BFT,Xtst_BFT,B,delta,lambda,K,Pre_Indx);

	%% Validation on the testing samples with the selected features
	% AUC
	Xfstrn = Xtrn_BFT(:, idx_fs);
	Xfstst = Xtst_BFT(:, idx_fs);
	[accREFS_tst, pREFS_tst] = validation(Xfstrn,Ytrn_BFT,Xfstst,Ytst_BFT,true);
	[~,~,~,AUCREFS_tst] = perfcurve(Ytst_BFT,pREFS_tst(:,2),2);
	disp(['test: acc ', num2str(mean(accREFS_tst)/100), ', auc ', num2str(AUCREFS_tst)])

	% F1-Score and Accuracy
	Preds = pREFS_tst(:,2);
	Yhat = 1.0*(Preds>0.5);
	stats = confusionmatStats(Ytst,Yhat);
	Accuracy = stats.accuracy(end); 
	F_Score = stats.Fscore(end);

	% print out Accuracy, AUC and F1-Score
	Final_Feat_Indices=Pre_Feat_Indices(idx_fs);
	disp('Selected features:')
	Final_Feats = miRNA_Names(Final_Feat_Indices);
	disp(Final_Feats)
	fprintf('%s+BFT-> Acc:%.4f AUC:%.4f F1:%.4f\n', Criteria{Cri_Indx}, Accuracy, AUCREFS_tst, F_Score);
	
	Perfs{Cri_Indx, 1} = Criteria{Cri_Indx};
	Perfs{Cri_Indx, 2} = Accuracy;
	Perfs{Cri_Indx, 3} = AUCREFS_tst;
	Perfs{Cri_Indx, 4} = F_Score;
end

for i = 1:length(Criteria)
    disp([Criteria{i},'+BFT->ACC:',num2str(Perfs{i,2}),',AUC:',num2str(Perfs{i,3}),',F1:',num2str(Perfs{i,4})]);
end
