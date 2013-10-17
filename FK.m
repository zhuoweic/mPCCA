function [EncodeMat]=FK(obj,features)
EncodeMat=zeros(1,2*obj.NDimensions*obj.NComponents);
P=posterior(obj,features);
%% computing fisher vector %%
P(find(P<1e-4)) = 0;
P = bsxfun(@rdivide,P,sum(P,2));
if ~isempty(features)
    for gmm_i=1:obj.NComponents
        EncodeMat(2*(gmm_i-1)*obj.NDimensions+1:(2*gmm_i)*obj.NDimensions)=...
            [(1/(size(features,1)*obj.PComponents(gmm_i)^0.5))*...
            sum(bsxfun(@times,P(:,gmm_i),bsxfun(@rdivide,bsxfun(@minus,features,obj.mu(gmm_i,:)),obj.Sigma(:,:,gmm_i))),1),...
            (1/(size(features,1)*(2*obj.PComponents(gmm_i))^0.5))*...
            sum(bsxfun(@times,P(:,gmm_i),bsxfun(@rdivide,bsxfun(@minus,features,obj.mu(gmm_i,:)).^2,(obj.Sigma(:,:,gmm_i).^2))-1),1)];
    end
end
%% power normalization %%
% EncodeMat=sign(EncodeMat).*(abs(EncodeMat).^alpha);
%% l2 normalization %%
% EncodeMat=EncodeMat./repmat(sum(EncodeMat.^2,2).^(0.5)+eps,[1,size(EncodeMat,2)]);