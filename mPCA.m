%MPCA Apply PCA for different components of Gaussian mixtures.

%% preparing data 'X'
%% 'x' is of the format:
%% dimFeature x numSamples

%% initialization %%
%% set up mixture model
ncentres = 2; 
input_dim = 2;
mix = gmm(input_dim, ncentres, 'diag');

%% initialize the mixture model using K-means
mix.centres = [0.2 0.8; 0.8, 0.2];
mix.covars = [0.01 0.01];
mix.projects = [1 1] ;

disp('Now we adapt the parameters of the mixture model iteratively using the')
disp('EM algorithm. Each cycle of the EM algorithm consists of an E-step')
disp('followed by an M-step.  We start with the E-step, which involves the')
disp('evaluation of the posterior probabilities (responsibilities) which the')
disp('two components have for each of the data points.')
disp(' ')
disp('Since we have labelled the two components using the colours red and')
disp('blue, a convenient way to indicate the value of a posterior')
disp('probability for a given data point is to colour the point using a')
disp('scale ranging from pure red (corresponding to a posterior probability')
disp('of 1.0 for the red component and 0.0 for the blue component) through')
disp('to pure blue.')
disp(' ')
disp('Press any key to see the result of applying the first E-step.')
pause;

%% initial E-step %%
post = gmmpost(mix, x);

disp('Next we perform the corresponding M-step. This involves replacing the')
disp('centres of the component Gaussians by the corresponding weighted means')
disp('of the data. Thus the centre of the red component is replaced by the')
disp('mean of the data set, in which each data point is weighted according to')
disp('the amount of red ink (corresponding to the responsibility of')
disp('component 1 for explaining that data point). The variances and mixing')
disp('proportions of the two components are similarly re-estimated.')
disp(' ')
disp('Press any key to see the result of applying the first M-step.')
pause;

%% initial M-step %%
options = foptions; 
options(14) = 1; % A single iteration
options(1) = 1; % display log likelihood error
mix = gmmem(mix, x, options);

%% initial projection matrix
disp('We can continue making alternate E and M steps until the changes in')
disp('the log likelihood at each cycle become sufficiently small.')
disp(' ')
disp('Press any key to see an animation of a further 9 EM cycles.')

%% Loop over EM iterations.
numiters = 9;
for n = 1 : numiters

	post = gmmpost(mix, x);
	[mix, options] = gmmem(mix, x, options);
	fprintf(1, 'Cycle %4d  Error %11.6f\n', n, options(8));
	
	%% update projection matrix
end
