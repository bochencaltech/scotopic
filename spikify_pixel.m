function X = spikify_pixel(p, param)
%SPIKIFY_PIXEL
% X = spikify_pixel(p, param)
% computes the *delta* photon counts at an array of exposure times for a
% ground truth image p. 
% If the exposure times are [10ms, 20ms, 50ms], then X contains the photon
% counts for the first 10ms, the second 10ms, and then 30ms. 
% p is in [0 255] raw pixel intensity, say from MNIST
%   p can be of arbitrary size. If p is [H W C], then 
%   X is [H W C N_bins ntrials]
%   where N_bins is the number of exposure times
% param contains
%   [Latest]:
%   .maxPPP:   white color
%   .PPPArray:  an array of test PPPs
%   [Deprecated]:
%   .maxT maximum exposure time (sec) [0.12]
%   .TArray     an array of expodure time at which an image is taken. 
%   .np_persec  expected number of photons per second for the brightest
%               color [1854]
%   .ntrials    number of repetitions [1]
%   -------------------------------
%   .snr        ratio between the brightest pixel to the darkest [30]
%   .snr_dc     ratio between PPP of the brightest pixel to variance of
%               dark current (Depricated!!)
%               snr already captures dark current, a snr=33 means
%               a dark current of 3%. 
% (c) 2016 Bo Chen
% bchen3@caltech.edu
if nargin == 0, timing_test; return; end
assert(min(p(:))>-0.01 & max(p(:))<256); % guard against mean subtraction
assert(~isfield(param, 'snr_dc')); % for historical reasons
assert(~isfield(param, 'ntrials') || param.ntrials == 1); % will work on this later. implementing ntrials > 1 may mess up dimensionality of data
siz = size(p);
if length(siz<3), siz = [siz ones(3-length(siz))]; end

white = param.maxPPP;
dark = white / param.snr;
N_photons_over_T = (p / 255)*(white - dark) + dark;
N_bins = length(param.PPPArray);
switch param.mode
  case {'batch_stochastic', 'batch'}
    
    X = nan([prod(siz), N_bins], 'single');
    Tinc = [param.PPPArray(1) diff(param.PPPArray)];
    for b = 1:N_bins 
        Pfire_per_dt = reshape( N_photons_over_T/param.maxPPP*Tinc(b), [1, 1, siz]);
        X(:,b) = poissrnd(Pfire_per_dt(:));
    end
    if isfield(param, 'jitter') && ~(param.jitter == 0)
        assert(isfield(param, 'input_size'));
        input_size = param.input_size;
        if length(input_size)==2, input_size = [input_size 1]; end
        batch_size = numel(X(:,1)) / prod(input_size);
        deg = 0; 
        for b = 1:N_bins
            im = reshape(X(:,b),[param.input_size batch_size]);
            % Amount of jitter is proportional to exposure time
            deg = deg + randn(batch_size,1)*sqrt(Tinc(b)*param.jitter);
            for ii=1:batch_size
                im(:,:,:,ii) = imrotate(im(:,:,:,ii), deg(ii),'bilinear','crop');
            end
            X(:,b) = im(:);
        end
    end
    if isfield(param, 'amp_sigma') && isfield(param, 'fpn_sigma') && ...
            ~(param.amp_sigma == 0 && param.fpn_sigma == 0)
        for b = 1:N_bins
            X(:,b) = corrupt(X(:,b), param.amp_sigma, param.fpn_sigma, Tinc(b)); 
        end
    end
    X = reshape(X, [siz N_bins]);
  case 'paired'
    % the last dimension of p (batchsize) must be the same as length(PPPArray)
    Pfire_per_dt = bsxfun(@times, reshape(N_photons_over_T, [],N_bins)/param.maxPPP, ...
                          reshape(param.PPPArray, 1, []));
    X = poissrnd(Pfire_per_dt(:));
    X = reshape(X, siz);
    if isfield(param, 'amp_sigma') && isfield(param, 'fpn_sigma') && ...
            ~(param.amp_sigma == 0 && param.fpn_sigma == 0) || ...
            isfield(param,'jitter')
        error('Not implemented, use batch or batch_stochastic mode.');
    end
  otherwise
    error('Unknown mode %s for spikify_pixel', param.mode);
end


end

function X = corrupt(X, amp_sigma, fpn_sigma, readtimes)
% X = corrupt(X, amp_sigma, fpn_sigma)
% current X by the amplier noise and the fixed pattern noise
% readtimes is the number of times the sensor is read (the sum of the
% multiple readings is the output signal)
% the noise sigmas are defined with respect to one read. Here
% we assume that on expectation one read produces 1 PPP.
    if amp_sigma > 0,
       X = X + amp_sigma*randn(size(X))*readtimes;
    end
    if fpn_sigma > 0,
       X = X .* (1+fpn_sigma*randn(size(X)));
    end

end


function timing_test
    p = rand(32,32,3,100)*255;
    param = struct('maxPPP',220, 'snr',33);
    param.PPPArray = [0.22, 2.2, 22, 220];
    param.mode = 'batch';
    tic;
    for i=1:10
        X = spikify_pixel(p, param);
        % X is [32 32 3 100 4]
        X = reshape(X, 32,32,3,400);
        ridx = randperm(400); ridx = ridx(1:100);
        X = X(:,:,:,ridx);
    end
    t1 = toc;
    fprintf('Batch mode time: %f\n', t1);
    
    param.PPPArray = repmat([0.22, 2.2, 22, 220], 1, 25);
    param.mode = 'paired';
    tic;
    for i=1:10
        Y = spikify_pixel(p, param);
    end
    t2 = toc;
    fprintf('Paired mode time: %f\n', t2);

end

