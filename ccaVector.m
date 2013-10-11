function gmModel = ccaVector(descrSIFT, descrGIST, alphaPower, diagV, sharedDim, numWords)
%CCAVECTOR a driver program which call MCCA and FK program to generate CCA vectors
%% codes: featureDim x numImages, encoded images as CCA vector codes

%% Extracting features (for TRAINING GMM models) %%
%% two kinds of feature extracted:
%% SIFT: descrSIFT, a matrix descriptors with columns as samples index
%% GIST: descrGIST, same as above

%MCCA a third-party program which train CCA mixture models for two kinds of feature
%% INPUT:
%% descrSIFT, descrGIST: two kinds of descriptors
%% sharedDim: dimension of their shared space
%% numWords: number of components as well as number of codewords
%%
%% OUTPUT:
%% transformX, transformY: transform matrix list from shared space to different feature representation
%% meanX, meanY: mean vector list for different kinds of feature corresponding to components
%% varX, varY: variance matrix list for two kinds of feature corresponding to components
%% weight: posterior weight for the different components

%% Training CCA mixture models %%
disp('***** training CCA mixture models *****') ;
[transformX, transformY, meanX, meanY, varX, varY, weight] = ...
	mcca(descrSIFT, descrGIST, sharedDim, numWords) ;
	
%% Generating Gaussian Mixture Models%%
disp('***** generating GMM models *****') ;
%% generate corresponding parameters for Gaussian components based on the trained parameters
%% denote the joint vector as V
dimX = size(meanX, 2) ;
dimY = size(meanY, 2) ;
%% meanV: dimV x numComponents
meanV = cat(2, meanX, meanY)' ;
dimV = size(meanV, 1) ;
%% varV: dimV x dimV x numComponents
varV = zeros(dimV, dimV, size(numWords)) ;
for indexComponent = 1 : numWords
	%% block matrix assignment
	%% left top
	varV(1 : dimX, 1 : dimX, indexComponent) = ...
		squeeze(transformX(indexComponent, :, :)) * squeeze(transformX(indexComponent, :, :))' + squeeze(varX(:, :, indexComponent)) ;
	%% right top
	varV(1 : dimX, 1 + dimX : dimY + dimX, indexComponent) = ...
		squeeze(transformX(indexComponent, :, :)) * squeeze(transformY(indexComponent, :, :))' ;
	%% left bottom
	%% transpose right top
	varV(1 + dimX : dimY + dimX, 1 : dimX, indexComponent) = ...
		varV(1 : dimX, 1 + dimX : dimY + dimX, indexComponent)' ;
	%% right bottom
	varV(1 + dimX  : dimY + dimX, 1 + dimX : dimY + dimX, indexComponent) = ...
		squeeze(transformY(indexComponent, :, :)) * squeeze(transformY(indexComponent, :, :))' + squeeze(varY(:, :, indexComponent)) ;
	%% reserve only the diagonal elements
	%% diagV: flag suggesting whether or not to only reserve the diagonal elements of variance matrix
	if diagV == true
		varV(:, :, indexComponent) = diag(diag(squeeze(varV(:, :, indexComponent)))) ;
	end
end
%% get the model
gmModel = gmdistribution(meanV, varV, weight') ;