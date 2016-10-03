function plot_sensor_noise_effect(RESFOLDER)
% plot_sensor_noise_effect(RESFOLDER)
% plot robustness analysis to noise
% (c) 2016 Bo Chen
% bchen3@caltech.edu
if ~exist('RESFOLDER', 'var')
    RESFOLDER = '.';
end
config = getScotopicConfig;

DATA = cell(1, 2);
DATA{1} = cell(1, 4);
mnist_folder = fullfile(config.data_folder, 'mnist-val_lrx0.004a0.00fwaldnet-wb1');
amp_sigma = 0.15; fpn_sigma=0.03; snr=0.97/0.03;
for experimentType = 1:4
    switch experimentType
        case 1
            vars = {'snr'};
            vals = {[0.97/0.03,20, 13, 10]};
            var2file = @(var1) sprintf('test_FR_nt50_s%.3f_a%.2f_f%.2f.mat',...
                var1,amp_sigma,fpn_sigma);
        case 2
            vars = {'amp_sigma'};
            vals = {[0.15, 0.22, 0.5]};
            var2file = @(var1) sprintf('test_FR_nt50_s%.3f_a%.2f_f%.2f.mat',...
                snr,var1,fpn_sigma);
        case 3
            vars = {'fpn_sigma'};
            vals = {[0.01 0.03 0.1]};
            var2file = @(var1) sprintf('test_FR_nt50_s%.3f_a%.2f_f%.2f.mat',...
                snr,amp_sigma,var1);
        case 4
            vars = {'jitter'};
            vals = {[0 0.01 0.1]};
            var2file = @(var1) sprintf('test_FR_nt50_j%.3f.mat',var1);
    end
    var = vars{1};
    valArray = vals{1}; val_N = length(valArray);
    DATA{1}{experimentType} = cell(1, val_N);
    for val_I=1:val_N
        aa = load(fullfile(mnist_folder, var2file(valArray(val_I)) ));
        %aa = load(sprintf('%stest_%s_nt52.mat',mnist_folder, var2file(valArray(val_I))));
        DATA{1}{experimentType}{val_I} = aa.info;
    end
end
%%
DATA{2} = cell(1,4);
CIFAR_folder = fullfile(config.data_folder, 'cifar-a1.00fwaldnet-wb1');
amp_sigma = 0.15; fpn_sigma=0.03; snr=0.97/0.03;
for experimentType = 1:4
    switch experimentType
        case 1
            vars = {'snr'};
            vals = {[0.97/0.03,20, 13, 10]};
            var2file = @(var1) sprintf('test_FR_nt50_s%.3f_a%.2f_f%.2f.mat',...
                var1,amp_sigma,fpn_sigma);
        case 2
            vars = {'amp_sigma'};
            vals = {[0.15, 0.22, 0.5]};
            var2file = @(var1) sprintf('test_FR_nt50_s%.3f_a%.2f_f%.2f.mat',...
                snr,var1,fpn_sigma);
        case 3
            vars = {'fpn_sigma'};
            vals = {[0.01 0.03 0.1]};
            var2file = @(var1) sprintf('test_FR_nt50_s%.3f_a%.2f_f%.2f.mat',...
                snr,amp_sigma,var1);
        case 4
            vars = {'jitter'};
            vals = {[0 0.01 0.1]};
            var2file = @(var1) sprintf('test_FR_nt50_j%.3f.mat',var1);
    end
    var = vars{1};
    valArray = vals{1}; val_N = length(valArray);
    DATA{2}{experimentType} = cell(1, val_N);
    for val_I=1:val_N
        aa = load(fullfile(CIFAR_folder, var2file(valArray(val_I)) ));
        DATA{2}{experimentType}{val_I} = aa.info;
    end
end

%%
b = Beautifier;
b.LW = 3; b.FS = 18; b.MSshape = 10;
datasetStr = {'MNIST', 'CIFAR10'};
MS = {'.','.','.','.'}; %{'>', 'o', 'd', '<'};

for ii = 1:2
    % get range
    ERrange = [inf, -inf];
    for experimentType = 1:length(DATA{ii}),
        for expID=1:val_N
            t=DATA{ii}{experimentType}{expID};
            ERrange(1) = min(ERrange(1), min(t.ER));
            ERrange(2) = max(ERrange(2), max(t.ER));
        end;
    end
    for experimentType = 1:length(DATA{ii})
        figure(ii*100+experimentType); clf;
        subplot('Position',[0.2, 0.2, 0.3, 0.3]);
        
        switch experimentType
            case 1
                var = 'snr';
                valArray = [0.97/0.03, 20, 13, 10]; val_N = length(valArray);
                lStr = cell(1, val_N);
                for val_I=1:val_N
                    lStr{val_I} = sprintf('\\epsilon_{dc}=%.2f',1/(valArray(val_I)-1));
                end
            case 2
                var = 'amp_sigma';
                valArray = [0.15, 0.22, 0.5]; val_N = length(valArray);
                lStr = cell(1, val_N);
                for val_I=1:val_N
                    lStr{val_I} = sprintf('\\sigma_{r}=%.2f',(valArray(val_I)));
                end
            case 3
                var = 'fpn_sigma';
                valArray = [0.01 0.03 0.1]; val_N = length(valArray);
                lStr = cell(1, val_N);
                for val_I=1:val_N
                    lStr{val_I} = sprintf('\\sigma_{fpn}=%.2f',(valArray(val_I)));
                end
            case 4
                var = 'jitter';
                valArray = [0 0.01 0.1]; val_N = length(valArray);
                lStr = cell(1, val_N);
                for val_I=1:val_N
                    lStr{val_I} = sprintf('\\sigma_{\\theta}=%.0f',ceil((valArray(val_I))*220));
                end
        end
        CLR = b.CLR(val_N);
        h = nan(1, val_N);
        
        for expID = 1:val_N
            t = DATA{ii}{experimentType}{expID};
            h(expID) = b.plot(t.mRT, t.ER, [MS{expID} '-'], 'Color', ...
                CLR(expID,:), 'MarkerFaceColor', 'w'); hold on;
        end
        if ii==1,
            xlim([.1, 4]);
            set(gca, 'XScale', 'log','XTick',[.22 2.2],'YGrid','on','YTick',[0.01 0.03 0.1]);
        else
            xlim([.1, 300]);
            set(gca, 'XScale', 'log','XTick',[.22 2.2 22 220],'YGrid','on','YTick',[0.3 0.45 0.6]);
        end
        b.xlabel('Median PPP'); b.ylabel('ER');
        if ii==1, b.legend(h,lStr); end
        b.title(sprintf('Sensitivity to %s', var)); grid on;
        
        if ii==1, set(gca, 'YScale', 'log');
        else set(gca, 'YScale', 'linear');
        end
        ylim(ERrange.*[0.95 1.05]);
        
    end
end

%%
OUTFILE = fullfile(RESFOLDER, 'sensor_noise');
for fig = [101:104 201:204]
    figure(fig);
    printplot(OUTFILE, 1);
end
printplot(OUTFILE, 3);
return;

