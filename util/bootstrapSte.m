% (c) 2015 Bo Chen
% bchen3@caltech.edu

function [s st p]= bootstrapSte(X, statfunc, groupsize, prc)
% estimate standard error from a vector X by breaking X down into multiple
% parts of (roughly) equal sizes, estimate stats on each, then compute the
% standard error using all the estimates.
% if gs=groupsize is provided, we assume that data are presented as gs
% chunks of equal size.
if exist('groupsize', 'var')
    chunkSize = groupsize;
    nChunk = floor(numel(X)/chunkSize);
    if~(nChunk*chunkSize==numel(X))
%        warning('groupsize fails to divide data size, cannot reliably associate data with runs'); 
    end
    X = reshape(X(1:chunkSize*nChunk), chunkSize, nChunk);
else

    X = X(:);
    n = length(X);
    nChunk = 10;
    chunkSize = floor(n/nChunk);
    ridx = randperm(nChunk*chunkSize);
    X = reshape(X(ridx), chunkSize, nChunk);
end
    st = statfunc(X);
%     st = nan(nChunk, 1);
%     for ci = 1:nChunk
%        st(ci) = statfunc(X(:, ci));
%     end

    s = std(st)/sqrt(nChunk);
    if exist('prc','var') % return percentile
        p = prctile(st, prc);
    end
end