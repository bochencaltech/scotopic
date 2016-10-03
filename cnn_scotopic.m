function [net, info,testinfo] = cnn_scotopic(varargin)
% CNN_SCOTOPIC   Demonstrates Scotopic Network on CIFAR-10 or MNIST
%
%
% Note: this code depends on a modified version of matconvnet
% modified code include: cnn_train.m, vl_simplenn.m
% (c) 2016 Bo Chen
% bchen3@caltech.edu

proj_dir = mfilename('fullpath');
config = getScotopicConfig;
data_dir = config.data_dir;
addpath(genpath(fullfile(proj_dir)));
%addpath(genpath(fullfile(proj_dir, '..')));
run(fullfile(fileparts(proj_dir), ...
    'matconvnet-1.0-beta18', 'matlab', 'vl_setupnn.m')) ;

opts.dataset = 'cifar';  % cifar or mnist
opts.validate = false;   % whether to use validation set
opts.modelType = 'waldnet-wb1' ;
opts.modelName = '';
% Define how data will be generated
opts.sp.maxPPP = 220;      % maximum PPP (if PPP=220, then m+std ~ 255)
opts.sp.PPPArray = [.22 2.2 22 220];  % array of simulated PPPs
opts.sp.snr = 0.97 /  0.03; % ratio between brightness level to darkness level
opts.sp.alpha =1; % strength of prior (used to smooth the input normalization, ...
% importance of this parameter should be low, as the
% alpha and beta should learn the correct depedence of
% the parameters on T.
opts.sp.mode = 'batch'; % paired (more efficient), batch_stochastic or batch mode
opts.thArray = log(logspace(log10(0.1+1e-4), log10(1-1e-4), 50));
opts.sp.testPPPArray = logspace(log10(0.22), log10(220), 50); % densely sample light levels for testing
opts.sp.amp_sigma = 0;
opts.sp.fpn_sigma = 0;
opts.sp.jitter = 0; % random rotation angle / PPP std
opts.sp.estimate_PPP = false;
opts.sp.disTh = [];

[opts, varargin] = vl_argparse(opts, varargin) ;

opts.expDir = fullfile(data_dir, ...
    sprintf('%s-%s%s', opts.dataset, opts.modelName, opts.modelType)) ;
% e.g. [data_dir]/cifar-lrx4waldnet-wb1
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = fullfile(data_dir, opts.dataset) ;
% e.g. [data_dir]/cifar
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
% e.g. [data_dir]/cifar-lrx4waldnet-wb1/imdb.mat
opts.whitenData = true ;                                % not used for mnist
opts.contrastNormalization = true ;                     % not used for mnist
opts.networkType = 'simplenn' ;
opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;

if opts.sp.estimate_PPP  % load PPP estimator
    load(fullfile([opts.dataset, '_PPPEstimator.mat']),'model');
    opts.sp.PPPEstimator = model;
end

opts.sp
if strcmp(opts.dataset,'mnist') && (opts.contrastNormalization || opts.whitenData)
    warning('Either contrastNormalization or whitening is set true, but they have no effect on MNIST\n');
end
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;
assert(ismember(opts.dataset, {'mnist', 'cifar'})); % dataset must be either cifar or mnist for now

% -------------------------------------------------------------------------
%                                                    Prepare model and data
% -------------------------------------------------------------------------
switch opts.modelType
    case 'lenet'
        net = cnn_scotopic_init('dataset', opts.dataset,'networkType', opts.networkType) ;
    case 'lenet-small'
        assert(strcmp(opts.dataset, 'cifar'));
        net = cnn_scotopic_init('dataset', opts.dataset,'networkType', opts.networkType, ...
            'size',[16 16 32 32]) ;
    case 'waldnet-1'
        net = cnn_scotopic_init('dataset', opts.dataset,'networkType', opts.networkType, ...
            'adaptLayer', 'first', 'sp', opts.sp) ;
    case 'waldnet-all'
        net = cnn_scotopic_init('dataset', opts.dataset,'networkType', opts.networkType,...
            'adaptLayer', 'all', 'sp', opts.sp) ;
    case 'waldnet-wb1'
        net = cnn_scotopic_init('dataset', opts.dataset,'networkType', opts.networkType,...
            'adaptLayer', 'wb-first', 'sp', opts.sp) ;
    case 'waldnet-wball'
        net = cnn_scotopic_init('dataset', opts.dataset,'networkType', opts.networkType,...
            'adaptLayer', 'wb-all', 'sp', opts.sp) ;
    case 'ensemble'
        opts.postfix = '';
        testinfo=cnn_ensemble(opts); net=[];info=[];
        return;
    case 'ensemble-small'
        opts.postfix = '-small';
        testinfo=cnn_ensemble(opts); net=[];info=[];
        return;
    case 'waldnet-wb1-large'
        assert(strcmp(opts.dataset, 'cifar'));
        net = cnn_scotopic_init('dataset', opts.dataset,'networkType', opts.networkType,...
            'adaptLayer', 'wb-first', 'sp', opts.sp, 'size',[96 96 96 64]) ;
    otherwise
        error('Unknown model type ''%s''.', opts.modelType) ;
end


if exist(opts.imdbPath, 'file')
    imdb = load(opts.imdbPath) ;
else
    switch opts.dataset
        case 'cifar'
            imdb = getCifarImdb(opts) ;
        case 'mnist'
            imdb = getMnistImdb(opts) ;
    end
    mkdir(opts.expDir) ;
    save(opts.imdbPath, '-struct', 'imdb') ;
end

net.meta.classes.name = imdb.meta.classes(:)' ;

% -------------------------------------------------------------------------
%                                                                     Train
% -------------------------------------------------------------------------

switch opts.networkType
    case 'simplenn', trainfn = @cnn_train ;
    case 'dagnn', trainfn = @cnn_train_dag ;
end

if opts.validate, use_set_id = 2; else use_set_id = 3; end

[net, info] = trainfn(net, imdb, getBatch(opts), ...
    'expDir', opts.expDir, ...
    net.meta.trainOpts, ...
    opts.train, ...
    'val', find(imdb.images.set == use_set_id)) ;

if opts.validate, net = load_test_net(opts.expDir, info); end
opts.sp.trainPPPArray = opts.sp.PPPArray;
opts.sp.PPPArray = opts.sp.testPPPArray;
opts.sp.mode = 'batch'; % change sampling mode to return all images under all light levels

%testinfo = cnn_scotopic_mnist_inceval(net, imdb, getBatch(opts), opts,  find(imdb.images.set==3)); return;
%testinfo = cnn_collect(net, getBatch(opts), opts); return;
testinfo.IT = cnn_test(net, imdb, getBatch(opts), opts, find(imdb.images.set==3)); % interrogation
testinfo.FR = SAT_constant_threshold(testinfo.IT.PPPArray, testinfo.IT.maxlogP_persample, ...
    testinfo.IT.acc_persample, opts.thArray, opts); % free response
testinfo.OFR = SAT_opt_thresohld(testinfo.IT, opts); % free response with optimized parameters

function net = load_test_net(expDir, info)
[~, mid] = min(info.val.error(1,:));
fprintf('Test using epoch %d\n', mid);
load(fullfile(expDir, sprintf('net-epoch-%d.mat', mid)), 'net');

function printSavefile(functionName, savefile)
fprintf('%s: saved results to\n%s\n', functionName, savefile);

function savefile = getSaveFile(opts, funcSig)
% funcSig: signature of function, used as follows:
%   test: ''
%   constant threshold FR: 'FR'
%   optimized threshold FR: 'OFR'
if ~isempty(funcSig), funcSig = [funcSig '_']; end
estStr = '';
if opts.sp.estimate_PPP, estStr = 'e_'; end

if opts.sp.amp_sigma == 0 && opts.sp.fpn_sigma==0 && opts.sp.jitter==0,
    savefile = fullfile(opts.expDir, ...
        sprintf('test_%s%snt%d.mat', ...
        funcSig, estStr, length(opts.sp.PPPArray)));
elseif opts.sp.jitter~=0
    savefile = fullfile(opts.expDir, ...
        sprintf('test_%s%snt%d_j%.3f.mat', ...
        funcSig, estStr, length(opts.sp.PPPArray),opts.sp.jitter));
else
    savefile = fullfile(opts.expDir, ...
        sprintf('test_%s%snt%d_s%.3f_a%.2f_f%.2f.mat',...
        funcSig, estStr, length(opts.sp.PPPArray),opts.sp.snr,opts.sp.amp_sigma,opts.sp.fpn_sigma))
end

function info = cnn_test(net, imdb, getBatch, opts, test)
savefile = getSaveFile(opts, '')
% filenames differ depending on whether to use PPP estimates
if exist(savefile, 'file'), load(savefile, 'info'); return; end
batchSize = 100;  % batchSize used for testing
net.layers{end} = struct('type','softmax');
testPPPArray = opts.sp.PPPArray;
ppp_N = length(testPPPArray)

interpmode = 'pchip';
f = @(x) log(x)
% adapt the network to test PPP
for l=1:length(net.layers)
    if strcmp(net.layers{l}.type, 'custom')
        switch net.layers{l}.original_type
            case 'bnorm'
                for ii=1:2
                    net.layers{l}.weights{ii} = interp1(f(opts.sp.trainPPPArray), ...
                        net.layers{l}.weights{ii}', f(testPPPArray), interpmode)';
                end
                net.layers{l}.weights{3} = permute(interp1(f(opts.sp.trainPPPArray), ...
                    permute(net.layers{l}.weights{3},[3 1 2]), f(testPPPArray), interpmode),...
                    [2 3 1]);
                net.layers{l}.PPPArray = testPPPArray;
        end
    end
end

% evaluation on validation set
% compute info
info.acc = nan(1, ppp_N);
info.acc_persample = nan(numel(test), ppp_N);
info.maxlogP_persample = nan(numel(test), ppp_N);
info.logP_persample = nan(10, numel(test), ppp_N);
info.PPPArray = opts.sp.PPPArray;
info.labels = nan(1, numel(test));

numBatches = ceil(numel(test)/batchSize);
[idxes, batches, allLogP, labels] = deal(cell(1, numBatches));
for i=1:numBatches
    idxes{i} = (i-1)*batchSize+1:min(i*batchSize, numel(test));
    batches{i} = test(idxes{i});
end

net.layers{end-1}.precious = true;
PPPEstimator = [];
if opts.sp.estimate_PPP, PPPEstimator = opts.sp.PPPEstimator; end
dataset = opts.dataset;
parfor i=1:numBatches
    %for i=1:numBatches
    t0 = clock;
    if isempty(PPPEstimator)
        [im, label] = getBatch(imdb, batches{i}) ;
        res = vl_simplenn(net, im, [], [], 'mode','test', 'conserveMemory', true);
    else % estimate PPP
        [im, label, raw] = getBatch(imdb, batches{i}) ;
        [~, PPP_I] = estimate_PPP(dataset, raw, PPPEstimator, testPPPArray); % force the estimates to be listed in testPPPArray
        res = vl_simplenn_estimate_PPP(net, im, PPP_I, [],[],'mode','test', 'conserveMemory', true);
    end
    batchSize = length(batches{i});
    labels{i} = label(1:batchSize);
    % compute prediction and errors
    logP = reshape(gather(res(end-1).x), 10, batchSize, ppp_N) ; % [1 x 1 C x NT]
    allLogP{i} = bsxfun(@minus, logP, logsumexp(logP, 1));
    
    fprintf('test: processed batch %3d of %3d in %.1f sec...\n', ...
        i, numBatches, etime(clock, t0)) ;
end

for i=1:numBatches
    logP = allLogP{i}; idx = idxes{i}; batchSize = length(batches{i});
    [max_logP, predictions] = max(logP, [], 1);  % predictions is [C, N T]
    info.acc_persample(idx, :) = bsxfun(@eq, predictions(1,:,:), reshape(labels{i}, 1, batchSize)) ;
    info.maxlogP_persample(idx,:) = reshape(max_logP, batchSize, ppp_N);
    info.logP_persample(:, idx, :) = logP;
    info.labels(1, idx) = labels{i};
end

info.acc = mean(info.acc_persample, 1);
save(savefile, 'info');
printSavefile('cnn_test',savefile);
% -------------------------------------------------------------------------
function res =  vl_simplenn_estimate_PPP(net, im, PPP_I, varargin)
for i=1:length(net.layers)
    if strcmp(net.layers{1}.type, 'custom')
        net.layers{1}.PPP_I = PPP_I;  % timed_Bnorm will see this and choose the right weights
    end
end
res = vl_simplenn(net, im, varargin{:});


% -------------------------------------------------------------------------
function info = SAT_constant_threshold(PPP_test, L, cc, thetaArray, opts)
savefile = getSaveFile(opts, 'FR');
if exist(savefile, 'file'), load(savefile, 'info'); return; end

[N, ppp_N] = size(cc);
assert(ppp_N == length(PPP_test));
th_N = length(thetaArray);

[rt, isCorrect] = deal(nan(N, th_N));

for th_I = 1:th_N
    [terminated, index] = max( L' > thetaArray(th_I) , ...
        [], 1 );
    index(~terminated) = ppp_N;
    rt(:,th_I) = PPP_test(index);
    isCorrect(:,th_I) = cc( sub2ind([N ppp_N], 1:N, index) );
end
info.mRT = reshape(median(rt,1), th_N, 1);
info.ER = reshape(1-mean(isCorrect,1), th_N, 1);
info.rt = rt;
info.isCorrect = isCorrect;
save(savefile, 'info');
printSavefile('SAT_consant_threshold',savefile);
% -------------------------------------------------------------------------
function testinfo = cnn_ensemble(opts)
disp(opts.expDir);
if ~exist(opts.expDir,'dir'), mkdir(opts.expDir); end
savefile =fullfile(opts.expDir, sprintf('test_nt%d.mat', ...
    length(opts.sp.testPPPArray)));
if ~exist(savefile, 'file')
    ppp_N=length(opts.sp.PPPArray);
    net = cell(1, ppp_N); info = cell(1, ppp_N); en = cell(1, ppp_N);
    for ppp_I = 1:ppp_N
        [net{ppp_I}, info{ppp_I},en{ppp_I}] = cnn_scotopic('dataset',opts.dataset,...
            'modelName', [opts.modelName, num2str(opts.sp.PPPArray(ppp_I))], ...
            'modelType',['lenet' opts.postfix],...
            'sp',struct('PPPArray',opts.sp.PPPArray(ppp_I), ...
                        'testPPPArray',opts.sp.testPPPArray),...
            'train',opts.train);
    end
    
    opts.sp.trainPPPArray = opts.sp.PPPArray;
    opts.sp.PPPArray = opts.sp.testPPPArray;
    testinfo.IT = en{1}.IT;
    testinfo.IT.PPPArray = opts.sp.PPPArray;
    [~, specialistIDArray] = min(abs(bsxfun(@minus, log(opts.sp.trainPPPArray(:)), log(opts.sp.PPPArray(:)'))), [], 1);
    
    for s_I = 1:max(specialistIDArray)
        ppp_IArray = (specialistIDArray == s_I);
        testinfo.IT.acc(ppp_IArray) = en{s_I}.IT.acc(ppp_IArray);
        testinfo.IT.acc_persample(:, ppp_IArray) = en{s_I}.IT.acc_persample(:,ppp_IArray);
        testinfo.IT.maxlogP_persample(:, ppp_IArray) = en{s_I}.IT.maxlogP_persample(:,ppp_IArray);
        testinfo.IT.logP_persample(:,:,ppp_IArray) = en{s_I}.IT.logP_persample(:,:,ppp_IArray);
    end
    
    info = testinfo.IT; save(savefile, 'info');
else
    t = load(savefile, 'info'); testinfo.IT = t.info;
    opts.sp.trainPPPArray = opts.sp.PPPArray;
    opts.sp.PPPArray = opts.sp.testPPPArray;
end

testinfo.FR = SAT_constant_threshold(testinfo.IT.PPPArray, testinfo.IT.maxlogP_persample, ...
    testinfo.IT.acc_persample, opts.thArray, opts); % free response
testinfo.OFR = SAT_opt_thresohld(testinfo.IT, opts); % free response with optimized parameters
% -------------------------------------------------------------------------
function info = SAT_opt_thresohld(IT, opts)
savefile = getSaveFile(opts, 'OFR');
if exist(savefile, 'file'), load(savefile, 'info'); return; end

F = @(logP) min( (logP - log(1-exp(logP))) / log(10), 20);
thCandidateArray = F([log(logspace(log10(0.09), log10(1), 500))]); thCandidateArray(end)=3;

[~, info.mRT, info.ER, info.rt, info.isCorrect] = sequential_threshold_solver(opts.sp.PPPArray, ...
    F(IT.maxlogP_persample), IT.acc_persample, ...
    'CTArray', logspace(-4, log10(3e-1), 10), ...
    'thetaArray', thCandidateArray, ...
    'ensemble_size', 10, ...
    'trainidx', 1:2000, 'validx', 3001:4000, 'testidx', 1:10000);
save(savefile, 'info');
printSavefile('SAT_opt_threshold',savefile);
% -------------------------------------------------------------------------
function fn = getBatch(opts)
% -------------------------------------------------------------------------
switch lower(opts.networkType)
    case 'simplenn'
        switch opts.dataset
            case 'cifar'
                fn = @(x,y,z) getScotopicSimpleNNCifarBatch(x,y,opts.sp) ;
            case 'mnist'
                fn = @(x,y,z) getScotopicSimpleNNMnistBatch(x,y,opts.sp) ;
        end
    case 'dagnn'
        error('DagNN not implemented for Scotopic WaldNet');
end

% -------------------------------------------------------------------------
function imdb = getCifarImdb(opts)
% -------------------------------------------------------------------------
% Preapre the imdb structure, returns image data with mean image subtracted
unpackPath = fullfile(opts.dataDir, 'cifar-10-batches-mat');
files = [arrayfun(@(n) sprintf('data_batch_%d.mat', n), 1:5, 'UniformOutput', false) ...
    {'test_batch.mat'}];
files = cellfun(@(fn) fullfile(unpackPath, fn), files, 'UniformOutput', false);
file_set = uint8([ones(1, 5), 3]);

if any(cellfun(@(fn) ~exist(fn, 'file'), files))
    url = 'http://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz' ;
    fprintf('downloading %s\n', url) ;
    untar(url, opts.dataDir) ;
end

data = cell(1, numel(files));
labels = cell(1, numel(files));
sets = cell(1, numel(files));
for fi = 1:numel(files)
    fd = load(files{fi}) ;
    data{fi} = permute(reshape(fd.data',32,32,3,[]),[2 1 3 4]) ;
    labels{fi} = fd.labels' + 1; % Index from 1
    sets{fi} = repmat(file_set(fi), size(labels{fi}));
end

set = cat(2, sets{:});
data = single(cat(4, data{:}));
imdb.images.data = data ; % use this to generate lowlight images

n_train = sum(set==1);
fprintf('Data: train(%d), val(%d), test(%d)\n', sum(set==1), sum(set==2), sum(set==3));
% remove mean in any case
dataMean = mean(data(:,:,:,set == 1), 4);
imdb.images.dataMean = dataMean;
data = bsxfun(@minus, data, dataMean);

% normalize by image mean and std as suggested in `An Analysis of
% Single-Layer Networks in Unsupervised Feature Learning` Adam
% Coates, Honglak Lee, Andrew Y. Ng

if opts.contrastNormalization
    fprintf('Contrast Normalization\n');
    z = reshape(data,[],size(data,4)) ;
    imdb.contrastNormalization.mean = mean(z,1);
    z = bsxfun(@minus, z, imdb.contrastNormalization.mean) ;
    n = std(z,0,1) ;
    imdb.contrastNormalization.contrast = n;
    z = bsxfun(@times, z, mean(n) ./ max(n, 40)) ;
    data = reshape(z, 32, 32, 3, []) ;
    
end

if opts.whitenData
    fprintf('Whitening\n');
    z = reshape(data(:,:,:,set==1),[],n_train) ;
    W = z*z'/n_train ;
    [V,D] = eig(W) ;
    % the scale is selected to approximately preserve the norm of W
    d2 = diag(D) ;
    en = sqrt(mean(d2)) ;
    imdb.whitenTransform = V*diag(en./max(sqrt(d2), 10))*V';
end

clNames = load(fullfile(unpackPath, 'batches.meta.mat'));

imdb.images.labels = single(cat(2, labels{:})) ;
imdb.images.set = set;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = clNames.label_names;

% -------------------------------------------------------------------------

% -------------------------------------------------------------------------
function [X, labels, raw] = getScotopicSimpleNNCifarBatch(imdb, batch, sp)
% sp.mode specifies how to organize data with different PPPs in a batch
% 'batch'
%   return N*ppp_N images, list all images in all PPPs
% 'batch_stochastic'
%   return N images, each image randomly gets assigned a random PPP
% 'paired'
%   return N images in total in chunks of size N/ppp_N, consective chunks
%   have the same PPP.
%
% -------------------------------------------------------------------------
N = length(batch);
input_size = [32 32 3]; d = prod(input_size);
X = imdb.images.data(:,:,:,batch); % this is raw
dcTransform.scaling = (sp.maxPPP-sp.maxPPP/sp.snr)/255;
dcTransform.offset = sp.maxPPP/sp.snr;
transformedMean = dcTransform.scaling*imdb.images.dataMean(:) + dcTransform.offset;
if sp.jitter>0, sp.input_size = input_size; end
if rand > 0.5, X=fliplr(X) ; end
switch sp.mode
    case {'batch', 'batch_stochastic'}
        Nbins = length(sp.PPPArray);
        % obtain data with serialized pixels
        X = cumsum(reshape(spikify_pixel(X(:), sp), d, N, Nbins), 3); % [H*W*C N sp.Nbins]
        if nargout>2, raw = reshape(X, [input_size N*Nbins]); end
        % divisive normalize X, then rescale to match raw data
        X = sp.maxPPP*bsxfun(@rdivide, bsxfun(@plus, X, transformedMean/sp.maxPPP*sp.alpha), ...
            reshape(sp.PPPArray + sp.alpha, [1, 1, Nbins]));
    case 'paired'
        Nbins = 1;
        sp.PPPArray = repmat(sp.PPPArray(:)', N/length(sp.PPPArray(:)), 1);
        sp.PPPArray = sp.PPPArray(:)';
        X = reshape(spikify_pixel(X, sp), d, N);
        if nargout>2, raw = reshape(X,[inputs_size N]); end
        X = sp.maxPPP*bsxfun(@rdivide, bsxfun(@plus, X, transformedMean/sp.maxPPP*sp.alpha), ...
            sp.PPPArray + sp.alpha);
end
% preprocessing:
% 1. remove pixelwise mean
X = bsxfun(@minus, X, transformedMean);
% [2]. contrast normalization
if isfield(imdb, 'contrastNormalization')
    local_mean = imdb.contrastNormalization.mean(batch) * dcTransform.scaling;
    local_std = max(40, imdb.contrastNormalization.contrast(batch)) * dcTransform.scaling;
    X = bsxfun(@rdivide, bsxfun(@minus, X, local_mean), local_std);
end
X = reshape(X, d, []);
% [3]. whitening transform
if isfield(imdb,'whitenTransform')
    X = imdb.whitenTransform * X;
end

% randomly sample N examples for efficiency...
switch sp.mode
    case 'batch_stochastic' % randomly sample N points
        sid = randperm(N*Nbins); sid = sid(1:N);
        X = reshape(X(:,sid), [input_size, N]);
        if nargout>2, raw = raw(:,:,:,sid); end
        % replicate the labels
        labels = repmat(imdb.images.labels(1,batch), 1, Nbins) ;
        labels = labels(1,sid);
    case 'batch'  % use full batch return N*Nbins points
        X = reshape(X, 32, 32, 3, []);
        labels = repmat(imdb.images.labels(1,batch), 1, Nbins) ;
    case 'paired' % return N points
        X = reshape(X, 32, 32, 3, N);
        labels = imdb.images.labels(1,batch);
end

% --------------------------------------------------------------------
function imdb = getMnistImdb(opts)
% --------------------------------------------------------------------
% Preapre the imdb structure, returns image data with mean image subtracted
files = {'train-images-idx3-ubyte', ...
    'train-labels-idx1-ubyte', ...
    't10k-images-idx3-ubyte', ...
    't10k-labels-idx1-ubyte'} ;

if ~exist(opts.dataDir, 'dir')
    mkdir(opts.dataDir) ;
end

for i=1:4
    if ~exist(fullfile(opts.dataDir, files{i}), 'file')
        url = sprintf('http://yann.lecun.com/exdb/mnist/%s.gz',files{i}) ;
        fprintf('downloading %s\n', url) ;
        gunzip(url, opts.dataDir) ;
    end
end

f=fopen(fullfile(opts.dataDir, 'train-images-idx3-ubyte'),'r') ;
x1=fread(f,inf,'uint8');
fclose(f) ;
x1=permute(reshape(x1(17:end),28,28,60e3),[2 1 3]) ;

f=fopen(fullfile(opts.dataDir, 't10k-images-idx3-ubyte'),'r') ;
x2=fread(f,inf,'uint8');
fclose(f) ;
x2=permute(reshape(x2(17:end),28,28,10e3),[2 1 3]) ;

f=fopen(fullfile(opts.dataDir, 'train-labels-idx1-ubyte'),'r') ;
y1=fread(f,inf,'uint8');
fclose(f) ;
y1=double(y1(9:end)')+1 ;

f=fopen(fullfile(opts.dataDir, 't10k-labels-idx1-ubyte'),'r') ;
y2=fread(f,inf,'uint8');
fclose(f) ;
y2=double(y2(9:end)')+1 ;

set = [ones(1,numel(y1)) 3*ones(1,numel(y2))];
data = single(reshape(cat(3, x1, x2),28,28,1,[]));
imdb.images.data = data ;


% original MNIST only has train (60k) / test (10k)
% resplit the data to make train (55k) / val (5k) / test (10k) sets
if opts.validate
    train_idx = find(set==1);
    set(train_idx(55001:end)) = 2; % use the last 5k train for val
end
n_train = sum(set==1);
fprintf('Data: train(%d), val(%d), test(%d)\n', sum(set==1), sum(set==2), sum(set==3));

dataMean = mean(data(:,:,:,set == 1), 4);
%data = bsxfun(@minus, data, dataMean) ;

imdb.images.dataMean = dataMean;
imdb.images.labels = cat(2, y1, y2) ;
imdb.images.set = set ;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),0:9,'uniformoutput',false) ;

% -------------------------------------------------------------------------
function [X, labels, raw] = getScotopicSimpleNNMnistBatch(imdb, batch, sp)
% sp.mode specifies how to organize data with different PPPs in a batch
% 'batch'
%   return N*ppp_N images, list all images in all PPPs
% 'batch_stochastic'
%   return N images, each image randomly gets assigned a random PPP
% 'paired'
%   return N images in total in chunks of size N/ppp_N, consective chunks
%   have the same PPP.
%
% -------------------------------------------------------------------------
N = length(batch);
input_shape = [28, 28, 1]; d = prod(input_shape);
X = imdb.images.data(:,:,:,batch);
dcTransform.scaling = (sp.maxPPP-sp.maxPPP/sp.snr)/255;
dcTransform.offset = sp.maxPPP/sp.snr;
transformedMean = dcTransform.scaling*imdb.images.dataMean(:) + dcTransform.offset;
if sp.jitter~=0, sp.input_size = input_shape; end

switch sp.mode
    case {'batch', 'batch_stochastic'}
        Nbins = length(sp.PPPArray);
        % obtain data with serialized pixels
        raw = cumsum(reshape(spikify_pixel(X(:), sp), d, N, Nbins), 3); % [H*W*C N sp.Nbins]
        % divisive normalize X, then rescale to match raw data
        X = sp.maxPPP*bsxfun(@rdivide, bsxfun(@plus, raw, transformedMean/sp.maxPPP*sp.alpha), ...
            reshape(sp.PPPArray + sp.alpha, [1, 1, Nbins]));
    case 'paired'
        Nbins = 1;
        sp.PPPArray = repmat(sp.PPPArray(:)', N/length(sp.PPPArray(:)), 1);
        sp.PPPArray = sp.PPPArray(:)';
        raw = reshape(spikify_pixel(X, sp), d, N);
        X = sp.maxPPP*bsxfun(@rdivide, bsxfun(@plus, raw, transformedMean/sp.maxPPP*sp.alpha), ...
            sp.PPPArray + sp.alpha);
end
if nargout>2, raw = reshape(raw, [input_shape numel(X)/d]); else raw = []; end
% preprocessing:
% 1. remove pixelwise mean
X = bsxfun(@minus, X, transformedMean);

% randomly sample N examples for efficiency...
switch sp.mode
    case 'batch_stochastic' % randomly sample N points
        sid = randperm(N*Nbins); sid = sid(1:N);
        X = reshape(X(:,sid), [input_shape numel(X)/d]);
        % replicate the labels
        labels = repmat(imdb.images.labels(1,batch), 1, Nbins) ;
        labels = labels(1,sid);
    case 'batch'  % use full batch return N*Nbins points
        X = reshape(X, [input_shape numel(X)/d]);
        labels = repmat(imdb.images.labels(1,batch), 1, Nbins) ;
    case 'paired' % return N points
        X = reshape(X, [input_shape, N]);
        labels = imdb.images.labels(1,batch);
end
