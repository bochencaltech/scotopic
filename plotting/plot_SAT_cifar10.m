function plot_SAT_cifar10(RESFOLDER)
% plot_SAT_cifar10(RESFOLDER)
% plot summary plots of scotopic classifiers on CIFAR10 to RESFOLDER
% (c) 2016 Bo Chen
% bchen3@caltech.edu
if ~exist('RESFOLDER', 'var')
    RESFOLDER = './';
end

config = getScotopicConfig;

folderArray = {'pensemble', 'ratelenet', 'p220lenet', 'a1.00fwaldnet-wb1'};
lStr = {'Ensemble', 'Rate', 'Photopic','WaldNet'};
f_N = length(folderArray); t = cell(f_N,3);
for f_I=1:f_N
t{f_I,1} = load(fullfile(config.data_dir, sprintf('cifar-%s', folderArray{f_N}), 'test_nt50.mat'));
t{f_I,2} = load(fullfile(config.data_dir, sprintf('cifar-%s', folderArray{f_N}), 'test_FR_nt50.mat'));
t{f_I,3} = load(fullfile(config.data_dir, sprintf('cifar-%s', folderArray{f_N}), 'test_OFR_nt50.mat'));
end

t{f_N+1,1} = load(fullfile(config.data_dir, sprintf('cifar-%s', folderArray{f_N}), 'test_e_nt50.mat'));
t{f_N+1,2} = load(fullfile(config.data_dir, sprintf('cifar-%s', folderArray{f_N}), 'test_FR_e_nt50.mat'));
t{f_N+1,3} = load(fullfile(config.data_dir, sprintf('cifar-%s', folderArray{f_N}), 'test_OFR_e_nt50.mat'));
f_N=f_N+1; lStr{end+1} = 'EstP';


%%
b = papermode(Beautifier);
b.MSdot = 25; b.MSshape = 5; b.LW = 4;

titleStr = {'Interrogation','FR (mean RT)','FR (median RT)', 'Optimized FR (mean RT)'};
t_N = length(lStr);

er0 = 1; for i =1:t_N, er0 = min(er0, 1-max(t{i}.info.acc)-0.01); end
for plot_method = 1:4
    figure(10 + plot_method); clf;
    subplot('Position',[0.1, 0.1, 0.4, 0.5],'Unit', 'Normalized');
%subplot(2,2, plot_method);
shapes = {'d','s','>','<','o','^','v'};
CLR = b.CLR(t_N); CLR = CLR(end:-1:1,:); 
GreenCLRid = t_N-3+1; CLR([end-1 GreenCLRid],:) = CLR([GreenCLRid end-1],:);
h = nan(1, t_N);

for i=1:t_N
    if plot_method == 1, 
        xx = t{i,1}.info.PPPArray; er = 1-t{i}.info.acc; 
        iw = 1-t{i,1}.info.acc_persample;
    elseif plot_method == 2,
        xx = mean(t{i,2}.info.rt); er = t{i,2}.info.ER;
        iw = 1-t{i,2}.info.isCorrect; rt = t{i,2}.info.rt;
    elseif plot_method == 3 || plot_method == 4
        if plot_method == 3, info = t{i,2}.info; xx = median(info.rt); er=info.ER;
        elseif plot_method==4, info=t{i,3}.info; xx = mean(info.rt); er=info.ER;
        else error('Unknown method'); end
        iw = 1-info.isCorrect; rt = info.rt;
    end
    
    h(i) = b.plot(xx, er, '.-', 'Color', CLR(i,:)); hold on;     
    Nt=length(xx); [ss, se] = deal(nan(1, Nt));
    for j=1:size(iw,2)
        if plot_method~=1
            ss(j) = bootstrapSte(rt(:,j),@mean);
            plot( [max(1e-1, xx(j)-ss(j)) xx(j)+ss(j)], er(j)*[1 1], 'Color', CLR(i,:), 'LineWidth',2);
        end
        se(j) = bootstrapSte(iw(:,j), @mean);
        plot( xx(j)*[1 1], [max(1e-3, er(j)-se(j)) er(j)+se(j)], 'Color', CLR(i,:), 'LineWidth',2);
    end
    
end
if plot_method==4, b.legend(h(:), lStr{:}); end %set(gca, 'YScale', 'log');
ylim([er0, 0.65]);
set(gca, 'XScale', 'log', ...
    'XTick',[0.22 2.2 22 220], ...
    'YTick', [0.2 0.4 0.6]); grid on;
xlim([0.15 300]); 
b.title(titleStr{plot_method}); 
b.xlabel('PPP'); b.ylabel('Error rate');
printplot(fullfile(RESFOLDER, ['SAT_cifar_' num2str(plot_method)]), 2);
end

%% Plot IT vs FRmean vs FRmedian
titleStr =  'FR vs INT';
figure(16); clf;
plot_method = 7;
i = 4; % index of WaldNet;
       CLR = brewermap(3,'set2');
er_des = 0.23;
[~, mid] = min(abs(1-t{i,1}.info.acc-er_des)); 
bb = linspace(-1, 2.5, 15);
p_IT = hist(log10(t{i,1}.info.PPPArray(mid)), bb); 
% find thresholds with 1% perf deg
[~, mid] = min(abs(er_des - t{i,2}.info.ER));
p_FR = hist(log10(t{i,2}.info.rt(:,mid)),bb);
[~, mid] = min(abs(er_des - t{i,3}.info.ER));
p_OFR = hist(log10(t{i,3}.info.rt(:,mid)),bb);
subplot('Position', [0.1, 0.1, 0.4, 0.22],'Unit','Normalized');
%ver_range = [0 4000];
%plot(log10(int_ppp)*[1 1],ver_range, '-', 'LineWidth',8,'Color', CLR(1,:)); hold on;
ver_range = [0 max([p_FR(:); p_OFR(:)])];
aa = bar(bb, [p_IT*ver_range(2); p_FR; p_OFR]', 0.95); hold on;
for j=1:3
    set(aa(j), 'FaceColor', CLR(j,:));
end

b.legend('Interrogation', 'FR','Optimized FR','Location','Best');
set(gca, 'FontSize', b.FS, 'XTick',log10([.22 2.2 22 220]), 'XTickLabel',{'.22','2.2','22','220'}); grid on;
ylim(ver_range); b.xlabel('PPP');b.ylabel('Counts'); xlim(log10([.1 300]));
printplot(fullfile(RESFOLDER, ['SAT_' num2str(plot_method)]), 2);

%% Plot improvements
PPP_test =  t{i,1}.info.PPPArray;
[OptER,RawER,Optss,Rawss] = deal(cell(1, t_N)); %RawER = cell(1, t_N);
for i=1:t_N
    for ii = [3 2]
    [aa,sid] = sort(mean(t{i,ii}.info.rt)); [aa uid] = unique(aa);
    id_use = sid(uid);
    ER = interp1(aa, t{i,ii}.info.ER(id_use), PPP_test);
    ss = nan(1, length(id_use));
    for j=1:length(id_use)
        ss(j) = bootstrapSte(t{i,ii}.info.isCorrect(:,id_use(j)), @mean);
    end
    ss = interp1(aa, ss, PPP_test);
    if ii==3
        Optss{i} = ss; OptER{i} = ER;
    else
        Rawss{i} = ss; RawER{i} = ER;
    end
    end
end
figure(18); clf;
plot_method = 8;
subplot('Position',[0.1, 0.1, 0.4, 0.22],'Unit', 'Normalized');
%subplot(2,2, plot_method);
CLR = b.CLR(t_N); CLR = CLR(end:-1:1,:); 
GreenCLRid = t_N-3+1; CLR([end GreenCLRid],:) = CLR([GreenCLRid end],:);

h = nan(1, t_N);
for i=1:t_N
    diffER = RawER{i} - OptER{i};
    diffss =  sqrt(Optss{i}.^2+Rawss{i}.^2);
    plot(PPP_test, diffER + diffss, 'Color', CLR(i,:)); hold on;
    plot(PPP_test, diffER - diffss, 'Color', CLR(i,:)); hold on;
    h(i) = b.plot(PPP_test, diffER, '-', 'Color', CLR(i,:)); hold on;
end
set(gca, 'XScale', 'log','XTick',[.22 2.2 22 220],'Xlim',[.1 300]); grid on;
b.xlabel('PPP'); b.ylabel('Risk Reduction'); ylim([-0.02, 0.2]);
%%
printplot(fullfile(RESFOLDER, ['SAT_' num2str(plot_method)]), 2);

