function [code]=fisherVector(obj, features)
%FISHERVECTOR
% obj: gmdistribution class initialized
% features: samples x dimFeature

%% initialization
code = zeros(1,2 * obj.NDimensions * obj.NComponents) ;
%% posterior gamma
P = posterior(obj, features) ;
%% ensure sparsity (better fit the assumption)
P(find(P < 1e-4)) = 0 ;
%% re-normalization for the sparse posterior
P = bsxfun(@rdivide, P, sum(P,2));

%% computing fisher vector %%
if ~isempty(features)
    for indexComponent = 1 : obj.NComponents
		code([1 : 2 * obj.NDimensions] + 2 * (indexComponent - 1) * obj.NDimensions) = ...
            [ ...
			(1 / (size(features, 1) * obj.PComponents(indexComponent) ^ 0.5)) * ... %% coefficient
            sum(bsxfun(@times, P(:,indexComponent), bsxfun(@rdivide, bsxfun(@minus, features, obj.mu(indexComponent, :)),obj.Sigma(:, :, indexComponent))), 1), ... %% mean term
            (1 / (size(features, 1) * (2 * obj.PComponents(indexComponent)) ^ 0.5)) * ... %% coefficient
            sum(bsxfun(@times, P(:,indexComponent), bsxfun(@rdivide, bsxfun(@minus, features, obj.mu(indexComponent, :)) .^ 2, (obj.Sigma(:, :, indexComponent) .^ 2)) - 1), 1) ... %% variance term
			];
    end
end

%% power normalization %%
% code = sign(code) .* (abs(code) .^ alpha) ;
%% l2 normalization %%
% code = code ./ repmat(sum(code .^ 2, 2) .^ 0.5 + eps, [1, size(code, 2)]) ;