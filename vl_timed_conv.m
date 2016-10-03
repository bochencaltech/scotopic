function res = vl_timed_conv(direction, res, dzdy, l)
% res = vl_timed_conv(direction, res, dzdy, l)
% (c) 2016 Bo Chen
% bchen3@caltech.edu

N = size(res.x, 4);

assert(ndims(res.x)>=3);


ppp_N = length(l.PPPArray);
miniN = N/ppp_N;
nout = size(l.weights{2},2);
switch direction
    case 'forward'
        res.x = vl_nnconv(res.x, l.weights{1}, [], ...
                          'pad', l.pad, ...
                          'stride', l.stride, ...
                          l.opts{:}) ;
        for ppp_I = 1:ppp_N
            res.x(:,:,:,(ppp_I-1)*miniN+1:ppp_I*miniN) = bsxfun(@plus, ...
                                    res.x(:,:,:,(ppp_I-1)*miniN+1:ppp_I*miniN), ...
                                    reshape(l.weights{2}(ppp_I,:),1, 1, nout));
        end
    case 'backward'
        % the following implementation is justified by tests/testGradConv.m
        [res.dzdx, res.dzdw{1}] = vl_nnconv(res.x, l.weights{1}, [], dzdy, ...
                          'pad', l.pad, ...
                          'stride', l.stride, ...
                          l.opts{:}) ;
        res.dzdw{2} = nan(ppp_N, nout, 'single');
        for ppp_I = 1:ppp_N
            res.dzdw{2}(ppp_I,:) = sum(sum(sum(...
                    dzdy(:,:,:,(ppp_I-1)*miniN+1:ppp_I*miniN), 1), 2), 4);
        end
end
