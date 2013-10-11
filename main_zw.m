clc;
clear all;
dbstop if error;
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
    addpath('/nfs/pengxj/code/my.toolbox/vlfeat-0.9.15/toolbox/mex/mexglx');
end

%% Initialization
alphaPower = 0.5 ;
diagV = 1 ;
sharedDim = 40 ;
numWords = 5 ;
numIteration = 2 ;

%% HMDB51 dataset evaluation
data_path = fullfile('data','HMDB51');
splitdir = fullfile('data','HMDB51_TestTrain_7030_splits');
%% Training Configuration
filenames = textread(['data', slash, 'HMDB_files.txt'], '%s');
for isplit = 1
    [train_fnames,test_fnames,saction]= get_HMDB_split(isplit,splitdir);
    sv_path = fullfile('res', 'HMDB51',['stip_sampled_features_HMDB_split' num2str(isplit), '.mat']);
    if ~exist(sv_path,'file')
        sampled_features = sampling_stip_HMDB(data_path,train_fnames, 100000);
        save(sv_path,'sampled_features');
    else
        load(sv_path);
    end
end
%% Training Samples:
%% HOG-sampled_features(:,1:72)
%% HOF-sampled_features(:,73:end)


%% Encoding & Classification
for isplit = 1 %:3
	%% prepare training and testing data
    [train_fnames,test_fnames,saction]= get_HMDB_split(isplit,splitdir);
    n_trclips = length(train_fnames)*length(train_fnames{1}); 
	n_tsclips = length(test_fnames)*length(test_fnames{1});
    n_clips = n_trclips + n_tsclips; 
	trainIndex = zeros(n_clips,1); 
	typeIndex = trainIndex;
	
    for ii = 1:length(train_fnames) %
        tmp = train_fnames{ii};
        for jj = 1:70
            fpath = fullfile(data_path,saction{ii}, strrep(tmp{jj},'.avi','.harris3d.stip.bin'));
            tr_fnames((ii-1)*70 + jj) = {fpath};
            trainIndex((ii-1)*70 + jj) = 1; typeIndex((ii-1)*70 + jj) = ii;
        end
    end
	
    for ii = 1:length(test_fnames) %
        tmp = test_fnames{ii};
        for jj = 1:30
            fpath = fullfile(data_path,saction{ii}, strrep(tmp{jj},'.avi','.harris3d.stip.bin'));
            ts_fnames((ii-1)*30 + jj) = {fpath}; typeIndex(n_trclips+(ii-1)*30 + jj) = ii;
        end
    end
    all_fnames = [tr_fnames,ts_fnames];
	
	featuresTrainedHOG = sampled_features(:,1:72)' ;
	numTransformHOG = 50 ;
	featuresTrainedHOF = sampled_features(:,73:end)' ;
	numTransformHOF = 50 ;
	%% preprocessing on the features
	disp('applying PCA on features') ;
	[transformHOG, ~, eigenHOG] = princomp(featuresTrainedHOG') ;
	transformHOG = transformHOG(:, 1 : numTransformHOG) ;
	principalHOG = transformHOG' * featuresTrainedHOG ;
	
	[transformHOF, ~, eigenHOF] = princomp(featuresTrainedHOF') ;
	transformHOF = transformHOF(:, 1 : numTransformHOF) ;
	principalHOF = transformHOF' * featuresTrainedHOF ;
	
	%% get Gaussian mixture model
	gmModel = ccaVector(principalHOG, principalHOF, alphaPower, diagV,sharedDim, numWords, numIteration) ;
	save('/nfs/zhuowei/gmModel.mat', 'gmModel') ;
	
	%% one fisher vector for each video
	allType_feas = {} ;
	for ii = 1:n_clips %
		fprintf('encoding video clip: %04d/%04d', ii, n_clips) ;
		fid = fopen(all_fnames{ii},'r');  
        fseek(fid,8,'bof');
		stip = fread(fid, [169,inf],'float');
		fclose(fid);
		
		%% Extracting features (for ENCODING) %%
		%% SIFT + GIST: jointFeature, dimV x samples
		featuresCodedHOG = transformHOG' * stip(8 : 79, :) ;
		featuresCodedHOF = transformHOF' * stip(80 :  169, :) ;
		jointFeature = [featuresCodedHOG ; featuresCodedHOF] ;
		
		%FK a third-party program which encodes every image given a Gaussian Mixture model
		allType_feas{ii} = FK(gmModel, jointFeature', alphaPower) ;
	end
	allType_feas = cat(1, allType_feas{:}) ;
	
	%% apply classification
	[cm,avg_acc] = classify_svm(allType_feas, trainIndex, typeIndex, 'KernelAverage', 'RBF'); 
	fprintf('average accuracy for split#%d: ', isplit) ;
	disp(avg_acc) ;
	acc_mats(isplit) = avg_acc;

end