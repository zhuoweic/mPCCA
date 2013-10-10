clc;
clear;
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
sharedDim = 60 ;
numWords = 256 ;

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

	%% get Gaussian mixture model
	gmModel = ccaVector(sampled_features(:,1:72)', sampled_features(:,73:end)', alphaPower, diagV,sharedDim, numWords) ;
	
	for ii = 1:n_clips %
		ii
		fid = fopen(all_fnames{ii},'r');  
        fseek(fid,8,'bof');
		stip = fread(fid, [169,inf],'float');
		fclose(fid);
		
		%% Extracting features (for ENCODING) %%
		%% SIFT + GIST: jointFeature, dimV x samples

		%FK a third-party program which encodes every image given a Gaussian Mixture model
		allType_feas = FK(gmModel, jointFeature', alphaPower) ;

		%% apply classification
		[cm,avg_acc] = classify_svm(allType_feas, trainIndex, typeIndex, 'KernelAverage', 'RBF'); 
		avg_acc
		acc_mats(isplit) = avg_acc;
	end
end