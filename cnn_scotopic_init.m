function net = cnn_scotopic_init(varargin)
% Initialize scotopic classifier
% (c) 2016 Bo Chen
% bchen3@caltech.edu

opts.dataset = 'cifar';
opts.networkType = 'simplenn' ;
opts.adaptLayer = 'none';
opts.sp = [];
opts = vl_argparse(opts, varargin) ;
switch opts.dataset
    case 'cifar'
        opts.size = [32 32 64 64]; 
        opts.lr = [.1 2];
    case 'mnist'
        opts.size = [20 50 500];
        opts.lr = [1 1];
    otherwise
        error('Unknown dataset %s', opts.dataset);
end
opts = vl_argparse(opts, varargin); % read opts.size again

switch opts.dataset
    case 'cifar'
        net = cifar_skeleton(opts.size, opts.lr);
        net.meta.trainOpts.learningRate = [0.05*ones(1,30) 0.005*ones(1,25) 0.0005*ones(1,20)];
        if strcmp(opts.adaptLayer, 'wb-first')
            net.meta.trainOpts.learningRate = net.meta.trainOpts.learningRate*4;
        end
        net.meta.trainOpts.weightDecay = 0.0001 ;
    case 'mnist'
        net = mnist_skeleton(opts.size, opts.lr);
        if strcmp(opts.adaptLayer, 'wb-first')
            net.meta.trainOpts.learningRate = 0.004*ones(1,80);% 4x learning rate in waldnet mode
        else
            net.meta.trainOpts.learningRate = 0.001*ones(1,80);
        end
end

net.meta.trainOpts.batchSize = 100 ;
net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate) ;

net = net_insertBNorm(net);

net = net_adapt(net, opts.adaptLayer, opts.sp);

% Fill in default values
net = vl_simplenn_tidy(net) ;

% Switch to DagNN if requested
switch lower(opts.networkType)
  case 'simplenn'
    % done
  case 'dagnn'
    net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
    net.addLayer('error', dagnn.Loss('loss', 'classerror'), ...
             {'prediction','label'}, 'error') ;
  otherwise
    assert(false) ;
end

function net = cifar_skeleton(siz, lr)

    % Define network CIFAR10-quick
    net.layers = {} ;

    % Block 1
    net.layers{end+1} = struct('type', 'conv', ...
                               'weights', {{0.01*randn(5,5,3,siz(1), 'single'), zeros(1, siz(1), 'single')}}, ...
                               'learningRate', lr, ...
                               'stride', 1, ...
                               'pad', 2) ;
    net.layers{end+1} = struct('type', 'pool', ...
                               'method', 'max', ...
                               'pool', [3 3], ...
                               'stride', 2, ...
                               'pad', [0 1 0 1]) ;
    net.layers{end+1} = struct('type', 'relu') ;

    % Block 2
    net.layers{end+1} = struct('type', 'conv', ...
                               'weights', {{0.05*randn(5,5,siz(1),siz(2), 'single'), zeros(1,siz(2),'single')}}, ...
                               'learningRate', lr, ...
                               'stride', 1, ...
                               'pad', 2) ;
    net.layers{end+1} = struct('type', 'relu') ;
    net.layers{end+1} = struct('type', 'pool', ...
                               'method', 'avg', ...
                               'pool', [3 3], ...
                               'stride', 2, ...
                               'pad', [0 1 0 1]) ; % Emulate caffe

    % Block 3
    net.layers{end+1} = struct('type', 'conv', ...
                               'weights', {{0.05*randn(5,5,siz(2),siz(3), 'single'), zeros(1,siz(3),'single')}}, ...
                               'learningRate', lr, ...
                               'stride', 1, ...
                               'pad', 2) ;
    net.layers{end+1} = struct('type', 'relu') ;
    net.layers{end+1} = struct('type', 'pool', ...
                               'method', 'avg', ...
                               'pool', [3 3], ...
                               'stride', 2, ...
                               'pad', [0 1 0 1]) ; % Emulate caffe

    % Block 4
    net.layers{end+1} = struct('type', 'conv', ...
                               'weights', {{0.05*randn(4,4,siz(3),siz(4), 'single'), zeros(1,siz(4),'single')}}, ...
                               'learningRate', lr, ...
                               'stride', 1, ...
                               'pad', 0) ;
    net.layers{end+1} = struct('type', 'relu') ;

    % Block 5
    net.layers{end+1} = struct('type', 'conv', ...
                               'weights', {{0.05*randn(1,1,siz(4),10, 'single'), zeros(1,10,'single')}}, ...
                               'learningRate', .1*lr, ...
                               'stride', 1, ...
                               'pad', 0) ;

    % Loss layer
    net.layers{end+1} = struct('type', 'softmaxloss') ;

    % Meta parameters
    net.meta.inputSize = [32 32 3] ;


function net = mnist_skeleton(siz, lr)
f = 0.01;
net.layers = {} ;
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(5,5,1,siz(1), 'single'), zeros(1, siz(1), 'single')}}, ...
                           'learningRate', lr, ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(5,5,siz(1),siz(2), 'single'),zeros(1,siz(2),'single')}}, ...
                           'learningRate', lr, ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(4,4,siz(2),siz(3), 'single'),  zeros(1,siz(3),'single')}}, ...
                           'learningRate', lr, ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(1,1,siz(3),10, 'single'), zeros(1,10,'single')}}, ...
                           'learningRate', lr, ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'softmaxloss') ;

    % Meta parameters
    net.meta.inputSize = [28 28 1] ;

function net = net_adapt(net, adaptLayer, sp)
switch adaptLayer
    case 'none' % done
    case 'first' % only adapt the first layer
        net.layers{1} = adaptConv(net.layers{1}, sp);
    case 'all' % adapt all layers
        for l=1:length(net.layers)
           if strcmp(net.layers{l}.type, 'conv')
               net.layers{l} = adaptConv(net.layers{l}, sp);
           end
        end
    case 'wb-first' % adapt the first layer, both weight and bias
        % batch normalization should take care of the scaling and offset
        net.layers{2} = adaptBnorm(net.layers{2}, sp);
    case 'wb-all' % adapt all layers
        for l=1:length(net.layers)
%             if strcmp(net.layers{l}.type, 'conv')
%                 net.layers{l}.learningRate(2) = 0;
%             else
            if strcmp(net.layers{l}.type, 'bnorm')
                net.layers{l} = adaptBnorm(net.layers{l}, sp);
            end
        end
    otherwise
        assert(false);
end

function l = adaptConv(l, sp)
    l.PPPArray = sp.PPPArray;
    ppp_N = length(l.PPPArray);
    nout = size(l.weights{2}, 2);
    l.type = 'custom';
    l.original_type = 'conv';
    l.forward = @(a, res1, res2, testMode) vl_timed_conv('forward', res1, [], a);
    l.backward = @(a, res1, res2) vl_timed_conv('backward',res1, res2.dzdx, a);
    l.weights{2} = zeros(ppp_N, nout, 'single');
    l.opts = {};

function l = adaptBnorm(l, sp)
    l.PPPArray = sp.PPPArray;
    ppp_N = length(l.PPPArray);
    nout = size(l.weights{1}, 1);
    l.type = 'custom';
    l.original_type = 'bnorm';
    l.forward = @(a, res1, res2, testMode) vl_timed_Bnorm('forward', res1, [], a, testMode);
    l.backward = @(a, res1, res2) vl_timed_Bnorm('backward',res1, res2.dzdx, a);
    l.weights{1} = ones(nout, ppp_N, 'single');
    l.weights{2} = zeros(nout, ppp_N, 'single');
    l.weights{3} = zeros(nout, 2, ppp_N, 'single');
    l.weightDecay = [0 0 0];
    

function net = net_insertBNorm(net)
    insertLayers = [];
    for l = 1:length(net.layers)
        if isfield(net.layers{l}, 'weights'), 
            insertLayers = cat(2, insertLayers, l); 
        end
    end
    k = 0;
    for l = insertLayers
        net = insertBnorm(net, l+k);
        k = k + 1;
    end


function net = insertBnorm(net, l)
% --------------------------------------------------------------------
assert(isfield(net.layers{l}, 'weights'));
% disable bias of the conv layer below (it will be modeled by BNorm)
net.layers{l}.weights{2}(:) = 0;
net.layers{l}.learningRate(2) = 0;

ndim = size(net.layers{l}.weights{1}, 4);
layer = struct('type', 'bnorm', ...
               'weights', {{ones(ndim, 1, 'single'), zeros(ndim, 1, 'single')}}, ...
               'learningRate', [1 1 0.05], ...
               'weightDecay', [0 0]) ;
net.layers{l}.biases = [] ;
net.layers = horzcat(net.layers(1:l), layer, net.layers(l+1:end)) ;

