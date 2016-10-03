function res = vl_timed_Bnorm(direction, res, dzdy, l, testMode)
% res = vl_timed_Bnorm(direction, res, dzdy, l, testMode)
% (c) 2016 Bo Chen
% bchen3@caltech.edu
  
N = size(res.x, 4);

assert(ndims(res.x)>=3);

ppp_N = length(l.PPPArray);
miniN = N/ppp_N;

idxArray = cell(1, ppp_N);
if ~isfield(l, 'PPP_I') || isempty(l.PPP_I)    
    for ppp_I = 1:ppp_N
        idxArray{ppp_I} = (ppp_I-1)*miniN+1:ppp_I*miniN;
    end
else
    % during test mode, the network may use PPP estimates instead of gt for
    % weight adaptation
    assert(testMode && numel(l.PPP_I) == N && max(l.PPP_I)<=ppp_N);
    for ppp_I = 1:ppp_N
        idxArray{ppp_I} = find(l.PPP_I==ppp_I);
    end
end
switch direction
    case 'forward'
       res.dzdw = [];
       for ppp_I=1:ppp_N
           idx = idxArray{ppp_I};
           if testMode
                res.x(:,:,:,idx) = vl_nnbnorm(res.x(:,:,:,idx),...
                    l.weights{1}(:,ppp_I), l.weights{2}(:,ppp_I), ...
                    'moments', l.weights{3}(:,:,ppp_I)) ;
           else
                res.x(:,:,:,idx) = vl_nnbnorm(res.x(:,:,:,idx),...
                    l.weights{1}(:,ppp_I), l.weights{2}(:,ppp_I));
           end
       end

    case 'backward'
        gpuMode = isa(res.x, 'gpuArray') ;
        res.dzdw = {nan(size(l.weights{1}),'single'), nan(size(l.weights{2}),'single'),...
            nan(size(l.weights{3}),'single')};
        res.dzdx = nan(size(res.x),'single');
        if gpuMode, 
            for i=1:3
                res.dzdw{i} = gpuArray(res.dzdw{i});
            end
            res.dzdx = gpuArray(res.dzdx);
        end
        for ppp_I=1:ppp_N
            idx = idxArray{ppp_I};
            [res.dzdx(:,:,:,idx), res.dzdw{1}(:,ppp_I), res.dzdw{2}(:,ppp_I), res.dzdw{3}(:,:,ppp_I)] = ...
              vl_nnbnorm(res.x(:,:,:,idx), l.weights{1}(:,ppp_I), l.weights{2}(:,ppp_I), dzdy(:,:,:,idx)) ;
                       
        end
        % multiply the moments update by the number of examples (N*ppp_N) in the batch
        % this is required to make the update additive for subbatches
        % and will eventually be normalized away 
        % see cnn_train.accumulate_gradient for update
        
        res.dzdw{3} = res.dzdw{3} * size(res.x,4)  ;

end
