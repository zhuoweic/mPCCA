clear;
% dbstop if error;
addpath('/nfs/zhuowei/matlibs/mPCCA') ;
cd /nfs/pengxj/code/DL_Encode
addpath(genpath('src'));

%% Linux and Windows
if ispc
    slash = '\';
    addpath('D:\my.toolbox');
    addpath('D:\my.toolbox\libsvm-3.14\matlab');
    addpath('D:\my.toolbox\vlfeat-0.9.15\toolbox\mex\mexw64');
else
    slash = '/';
    addpath('/nfs/pengxj/code/my.toolbox');
    addpath('/nfs/pengxj/code/my.toolbox/libsvm-3.14/matlab');
    run('/nfs/pengxj/code/my.toolbox/vlfeat-0.9.15/toolbox/vl_setup');
	run('/nfs/zhuowei/matlibs/vlfeat-0.9.17/toolbox/vl_setup.m') ;
    addpath('/nfs/pengxj/code/my.toolbox/vlfeat-0.9.15/toolbox/mex/mexglx');
end

%% Initialization
alphaPower = 0.5 ;
diagV = 1 ;
sharedDim = 45 ;
%% number of components
numWords = 256 ;
numIteration = 100 ;
numFeaturesPerWord = 10000 ;
%% PCA dimensions
numTransformHOG = 48 ;
numTransformHOF = 54 ;
%% coding method
codingMethod = 'fv' ;
% codingMethod = 'cca' ;
%% fusion method
% fusionMethod = 'early' ;
fusionMethod = 'late' ;
%% features used
featureEmployed = [1 0] ; % use HOG only
%% HMDB51 dataset preparation
data_path = fullfile('data','HMDB51');
splitdir = fullfile('data','HMDB51_TestTrain_7030_splits');
filenames = textread(['data', slash, 'HMDB_files.txt'], '%s');
classRange = 51 ;

%% Encoding & Classification
for isplit = 1 : 3
	fprintf('***** processing dataset: %02d/%02d *****\n', isplit, 3) ;
	%% reading training clips
    [train_fnames,test_fnames,saction]= get_HMDB_split(isplit,splitdir);
    sv_path = fullfile('res', 'HMDB51',['stip_sampled_features_HMDB_split' num2str(isplit), '.mat']);
    if ~exist(sv_path,'file')
        sampled_features = sampling_stip_HMDB(data_path,train_fnames, numFeaturesPerWord * numWords);
        save(sv_path,'sampled_features');
    else
        load(sv_path);
    end
	%% split training features
	featuresTrainedHOG = sampled_features(:,1:72)' ;
	featuresTrainedHOF = sampled_features(:,73:end)' ;
	%% preprocessing on the features
	disp('***** applying PCA on features *****') ;
	%% HOG-sampled_features(:,1:72)
	[transformHOG, ~, eigenHOG] = princomp(featuresTrainedHOG') ;
	transformHOG = transformHOG(:, 1 : numTransformHOG) ;
	%% PCA-HOG
	principalHOG = transformHOG' * featuresTrainedHOG ;
	%% HOF-sampled_features(:, 73 : end)
	[transformHOF, ~, eigenHOF] = princomp(featuresTrainedHOF') ;
	transformHOF = transformHOF(:, 1 : numTransformHOF) ;
	%% PCA-HOF
	principalHOF = transformHOF' * featuresTrainedHOF ;
	
	%% prepare training and testing ground truth
    [train_fnames,test_fnames,saction]= get_HMDB_split(isplit,splitdir);
    n_trclips = length(train_fnames)*length(train_fnames{1}); 
	n_tsclips = length(test_fnames)*length(test_fnames{1});
    n_clips = n_trclips + n_tsclips; 
	trainIndex = zeros(n_clips,1); 
	typeIndex = trainIndex;
	%% configuring ground truth: training
    for ii = 1:length(train_fnames) %
        tmp = train_fnames{ii};
        for jj = 1:70
            fpath = fullfile(data_path,saction{ii}, strrep(tmp{jj},'.avi','.harris3d.stip.bin'));
            tr_fnames((ii-1)*70 + jj) = {fpath};
            trainIndex((ii-1)*70 + jj) = 1; typeIndex((ii-1)*70 + jj) = ii;
        end
    end
	%% configuring ground truth: testing	
    for ii = 1:length(test_fnames) %
        tmp = test_fnames{ii};
        for jj = 1:30
            fpath = fullfile(data_path,saction{ii}, strrep(tmp{jj},'.avi','.harris3d.stip.bin'));
            ts_fnames((ii-1)*30 + jj) = {fpath}; typeIndex(n_trclips+(ii-1)*30 + jj) = ii;
        end
    end
    all_fnames = [tr_fnames,ts_fnames];
		
	disp('***** training GMM *****') ;
	%% get Gaussian mixture model
	if (strcmp(codingMethod, 'cca') && strcmp(fusionMethod, 'early'))
		disp('***** using early CCA model *****') ;
		%% CCA
		[meanV, diagVarV, weight] = ccaVector(principalHOG, principalHOF, alphaPower, diagV, sharedDim, numWords, numIteration) ;
	elseif (strcmp(codingMethod, 'fv') && strcmp(fusionMethod, 'early'))
		%% fisher vector modeling
		disp('***** using early FV model *****')
		[meanV, diagVarV, weight] = vl_gmm([principalHOG ; principalHOF], numWords, 'verbose', 'MaxNumIterations', numIteration) ;
	elseif (strcmp(codingMethod, 'fv') && strcmp(fusionMethod, 'late'))
		disp('***** using late FV model *****')
		[meanHOG, diagVarHOG, weightHOG] = vl_gmm(principalHOG, numWords, 'verbose', 'MaxNumIterations', numIteration) ;
		[meanHOF, diagVarHOF, weightHOF] = vl_gmm(principalHOF, numWords, 'verbose', 'MaxNumIterations', numIteration) ;
	end

	%% saving gaussian mixture model
	% save('/nfs/zhuowei/gmModel.mat', 'gmModel') ;
	
	%% one fisher vector for each video
	allType_feas = ones(2 * numWords * (numTransformHOG + numTransformHOF), n_clips) ;
	%% encode images
	disp('***** encoding images *****') ;
	parfor ii = 1:n_clips %
		% fprintf('encoding video clip: %04d/%04d\n', ii, n_clips) ;
		fid = fopen(all_fnames{ii},'r');  
        fseek(fid,8,'bof');
		stip = fread(fid, [169,inf],'float');
		fclose(fid);
		
		%% checking validity of STIP
		if size(stip, 1) == 169
			%% Extracting features (for ENCODING) %%
			featuresCodedHOG = transformHOG' * stip(8 : 79, :) ;
			featuresCodedHOF = transformHOF' * stip(80 :  169, :) ;
			if strcmp(fusionMethod, 'early')
				%% SIFT + GIST: jointFeature, dimV x samples
				jointFeature = [featuresCodedHOG ; featuresCodedHOF] ;			
				%% using original fisher vector
				% allType_feas(:, ii) = fisherVector(gmdistribution(meanV', reshape(diagVarV, [1 size(diagVarV)]), weight), jointFeature') ;
				%% using improved fisher vector
				allType_feas(:, ii) = vl_fisher(jointFeature, meanV, diagVarV, weight, 'Improved') ;
			elseif strcmp(fusionMethod, 'late')
				fvHOG = vl_fisher(featuresCodedHOG, meanHOG, diagVarHOG, weightHOG, 'Improved') ;
				fvHOF = vl_fisher(featuresCodedHOF, meanHOF, diagVarHOF, weightHOF, 'Improved') ;
				allType_feas(:, ii) = [fvHOG ; fvHOF] ;
			end
		else
			%% some missing data
			%% using uniform distribution
			allType_feas(:, ii) = allType_feas(:, ii) / (2 * numWords * (numTransformHOG + numTransformHOF)) ;
		end
	end
	% allType_feas = cat(1, allType_feas{:}) ;

	%% saving codes
	% save('/nfs/zhuowei/actionCodes.mat', 'allType_feas') ;

	%% split train/test dataset
	train = find(trainIndex == 1) ;
	test = find(trainIndex == 0) ;

	%% SVM classification %%
	%% parameters initialization
	C = 100 ;
	lambda = 1 / (C * numel(train)) ;
	par = {'Solver', 'sdca', 'Verbose', ...				
		   'BiasMultiplier', 1, ...
		   'Epsilon', 0.001, ...
		   'MaxNumIterations', 100 * numel(train)} ;

	scores = cell(1, classRange) ;	% predict scores
	ap = zeros(1, classRange) ;		% average precision
	w = cell(1, classRange) ;		% svm weight
	b = cell(1, classRange) ;		% svm shift
	%% apply classification	
	for classIndex = 1 : classRange
		fprintf('classifying type: %02d/%02d\n', classIndex, classRange) ;
		classLabel = 2 * (typeIndex == classIndex) - 1 ;
		[w{classIndex},b{classIndex}] = vl_svmtrain(allType_feas(:,train), classLabel(train), lambda, par{:}) ;
		scores{classIndex} = transpose(w{classIndex}) * allType_feas + b{classIndex} ;

		%% computing precision-recall curve
		%% note the old standard of average precision calculation method: ap11
		[~,~,info] = vl_pr(classLabel(test), scores{classIndex}(test)) ;
		ap(classIndex) = info.ap ;
	end
	scores = cat(1,scores{:}) ;

	% [cm,avg_acc] = svmZW(allType_feas', trainIndex, typeIndex, 'KernelAverage', 'RBF'); 
	[~,preds] = max(scores, [], 1) ;
	confusion = zeros(classRange) ;
	for classIndex = 1 : classRange
		sel = find(typeIndex == classIndex & trainIndex == 0) ;
		tmp = accumarray(preds(sel)', 1, [classRange 1]) ;
		tmp = tmp / max(sum(tmp),1e-10) ;
		confusion(classIndex,:) = tmp(:)' ;
	end
	
	avgAccuracy = mean(diag(confusion)) ;
	fprintf('average accuracy for split#%d: ', isplit) ;
	disp(avgAccuracy) ;
	acc_mats(isplit) = avgAccuracy ;
end

disp('Average Performance on HMDB51: ') ;
disp(mean(acc_mats)) ;
disp(acc_mats) ;