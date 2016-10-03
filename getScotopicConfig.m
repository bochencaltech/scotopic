function config = getScotopicConfig
% Configuration files for the code base. 
% Customize for your workstation
% (c) 2016 Bo Chen
% bchen3@caltech.edu

% where to store data, must have write permission
config.data_dir = '/Users/bochen/Data/scotopic';
% '/scratch/bochen3/matconvnet/scotopic_release';
if ~exist(config.data_dir, 'dir')
    succ = mkdir(config.data_dir);
    if ~succ, 
        error('Data dir %s does not exist and cannot be created!\n', config.data_dir);
    end
end

% whether to use GPU
config.use_gpu = false;