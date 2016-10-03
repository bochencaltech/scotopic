function [PPP, PPP_I]= estimate_PPP(dataset, rawim, model, PPPArray)
% [PPP, PPP_I]= estimate_PPP(dataset, rawim, model, PPPArray)
% estimate PPP from raw images rawim.
% if PPPArray is provided, enforce the PPP estimates to be
% elements within PPPArray, and provide the index PPP_I
% (c) 2016 Bo Chen
% bchen3@caltech.edu

assert(ismember(model.type, model.allTypes));
[nx, ny, nc, N] = size(rawim);
switch dataset
    case 'mnist'
        input_size = [28 28 1];
        assert(all(input_size==[nx ny nc]));
        
        switch model.type
            case 'max_resp'
                assert(strcmp(model.convParam.pool,'max'));
                a = squeeze(max(max(convn(rawim, model.convParam.weights),[],1),[],2));
                PPP = model.predFunc(model.predParam, double(a));
            case 'avg_prc'
                a = sort(reshape(rawim, prod(input_size), []), 1, 'descend');
                a = mean(a(1:model.topK,:));
                PPP = model.predFunc(model.predParam, double(a));
        end
    case 'cifar'
        input_size = [32 32 3];
        assert(all(input_size==[nx ny nc]));
        switch model.type
            case 'gray_max_resp'
                assert(strcmp(model.convParam.pool,'max'));
                a = squeeze(max(max(convn(rawim, model.convParam.weights),[],1),[],2));
                PPP = model.predFunc(model.predParam, double(a));
            case 'resp_prc'
                
                a = convn(rawim, ones(model.S, model.S, 1, 1)/model.S^2);
                a = sort(reshape(a, [], N), 1, 'descend');
                a = median(a(1:model.topK,:));
                PPP = model.predFunc(model.predParam, double(a));
        end
end
PPP_I = nan(N,1);
if exist('PPPArray','var'),
    [logPPP, PPP_I] = round_to_targets(log(PPP), ...
        log(PPPArray));
    PPP = exp(logPPP);
end
end

function [vRounded, Index] = round_to_targets(v, roundTargets)


[v, sid] = sort(v,'ascend');
N_bins = numel(roundTargets);
[~,Index1] = histc(v,[-Inf interp1(1:N_bins,roundTargets, ...
    0.5 + (1:N_bins-1)) Inf]);
Index1 = min(Index1, N_bins);
vRounded1 = roundTargets(Index1);
vRounded = nan(size(vRounded1));
Index = nan(size(Index1));
vRounded(sid) = vRounded1;
Index(sid) = Index1;
end