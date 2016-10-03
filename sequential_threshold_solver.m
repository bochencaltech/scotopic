function [theta, mRT, ER, rt, isCorrect] = sequential_threshold_solver(TArray, L, cc, varargin)
% [theta, mRT, ER, rt, isCorrect] = sequential_threshold_solver(TArray, L, cc, varargin)
% train a set of thresholds over time to maximize Bayesrisk.
%
% TArray: [1 x Nt] an array of time points
% L:    [N, Nt] max log likelihood ratios
% cc:   [N, Nt] whether the categorical prediction is correct
% optional parameters:
%   .trainidx: [1 x ntr] indices of training data [first 10% of data]
%   .testidx:  [1 x nts] indices of test daata
%   .CTArray:  [1 x CT_N] array of time costs in 1 / second
%   [for ensemble method only]
%   .ensemble_size: scalar, how many models to use to estimate threshold
%   .thetaArray: [1 x theta_N] an array of values to search for the threshold's
%   value
%
% return:
%   theta: [1 x Nt] array of thresholds
%   mRT:   [CT_N x 1] median RT
%   ER:    [CT_N x 1] error rates
%   rt:    [N x CT_N] and isCorrect: [ N x CT_N]
%           rt and indicator for correct detection per data and cost of time
% (c) 2016 Bo Chen
% bchen3@caltech.edu


if nargin == 0, plotall(); return; end

[N, Nt] = size(L);
assert(length(TArray) == Nt);
opts.trainidx = 1:round(N*0.1);
opts.validx = round(N*0.1)+1:round(N*0.2);
opts.testidx = round(N*0.2)+1:N;
opts.CTArray = [1e-5 1e-4 1e-3 1e-2 2e-2 5e-2 0.1 0.2 0.5 1];
opts.ensemble_size = 10;
opts.thetaArray = [log(logspace(log10(0.09), log10(1.01), 500)) 0];

opts = vl_argparse(opts, varargin);


CT_N = length(opts.CTArray);
CE = 1; theta = nan(CT_N, Nt);

[mRT, ER] = deal(nan(CT_N, 1));
[rt, isCorrect] = deal(nan(length(opts.testidx), CT_N));

tr.L = L(opts.trainidx,:);
tr.cc = cc(opts.trainidx,:);
ts.L = L(opts.testidx,:);
ts.cc = cc(opts.testidx,:);
val.L = L(opts.validx,:);
val.cc = cc(opts.validx,:);

for CT_I = 1:CT_N
    tic;
    CT = opts.CTArray(CT_I);
    % use model average to seed to initial solution for gradient descent
    this_theta = nan(Nt, opts.ensemble_size);
    batch_size = floor(length(opts.trainidx) / opts.ensemble_size);
    parfor i=1:opts.ensemble_size
        idx = (i-1)*batch_size+1:i*batch_size;
        this_theta(:,i) = solve_for_sequential_thresholds(opts.thetaArray, TArray,tr.L( idx,:),tr.cc(idx,:),CE,CT);
    end
    theta_en = mean(this_theta, 2);
    
    toc;
    
    figure(10); clf;
    subplot(121);
    sthe = std(this_theta, 0, 2);
    plot(TArray, theta_en); hold on;
    plot(TArray, theta_en+sthe, 'b--');
    plot(TArray, theta_en-sthe, 'b--');
    [S.m1, S.s1, S.m0, S.s0] = deal(nan(1, Nt));
    for t_I=1:Nt
        S.m1(t_I) = mean(tr.L(tr.cc(:,t_I)==1, t_I));
        S.s1(t_I) = std(tr.L(tr.cc(:,t_I)==1, t_I));
        S.m0(t_I) = mean(tr.L(tr.cc(:,t_I)==0, t_I));
        S.s0(t_I) = std(tr.L(tr.cc(:,t_I)==0, t_I));
    end
    plot(TArray, S.m1, 'g-');
    plot(TArray, S.m1+S.s1, 'g--');plot(TArray, S.m1-S.s1, 'g--');
    plot(TArray, S.m0, 'm-');
    plot(TArray, S.m0+S.s0, 'm--');plot(TArray, S.m0-S.s0, 'm--');
    
    set(gca, 'XScale', 'log');
    xlim([0.22 220]);
    drawnow; pause(0.1);
    
    
    theta(CT_I,:) = theta_en;
    
    [~, mRT(CT_I), ER(CT_I), rt(:,CT_I), isCorrect(:,CT_I)]= eval_sequential_threshold_cost(theta(CT_I,:),TArray,ts.L,ts.cc,CE,CT);
    [theta(CT_I,:), temp_cost, alltheta] = iterative_solve_for_sequential_thresholds(theta(CT_I,:), opts.thetaArray, TArray,tr.L, tr.cc,CE,CT);
    [trcost, tr_softcost,valcost,val_softcost] = deal(nan(size(temp_cost)));
    
    for ii = 1:3:length(alltheta)
        [valcost(ii), ~, ~, ~, ~, ~, val_softcost(ii)] = eval_sequential_threshold_cost(alltheta{ii},TArray,val.L,val.cc,CE,CT);
    end
    fprintf('CT_I=%d/%d:  %.3fsecs, ', CT_I, CT_N, toc);
    [mvalcost, mid] = nanmin(valcost);
    fprintf('valcost: %.3f, mid=[%d]\n', mvalcost, mid);
    figure(10); subplot(121);
    semilogx(TArray, alltheta{mid}, 'r-', 'LineWidth', 2); xlim([0. 220]);
    drawnow; pause(0.2);
    theta(CT_I,:) = alltheta{mid};
    [~, mRT(CT_I), ER(CT_I), rt(:,CT_I), isCorrect(:,CT_I)]= eval_sequential_threshold_cost(theta(CT_I,:),TArray,ts.L,ts.cc,CE,CT);
    [mean(rt(:,CT_I)) ER(CT_I)]
    
end

end


function [cost, grad] = sequential_threshold_cost_wrapper(theta, TArray, L, cc, CE, CT, T)
[~, ~,~,~,~, grad, cost] = eval_sequential_threshold_cost(theta, TArray, L, cc, CE, CT, T);
end

function [cost, mRT, ER, rt, isCorrect, semigrad, semicost] = eval_sequential_threshold_cost(theta, TArray, L, cc, CE, CT, T)

Nt = length(TArray); N = size(L,1);
assert(all(size(theta)==[1, Nt]));
[terminated, index] = max( bsxfun(@ge, L , theta), ...
    [], 2 );
index(~terminated) = Nt;
rt = TArray(index);
isCorrect = cc(sub2ind([N Nt], 1:N, index'));

cost = median(rt)*CT + mean(1-isCorrect)*CE;
mRT = median(rt);
ER = mean(1-isCorrect);

if nargout > 5,  % need return semigrad
    if ~exist('T','var'), T = 0.01; end
    dt = [TArray(1) diff(TArray)];
    sigmoid = @(x) 1./(1 + exp(-x));
    pass_through = sigmoid(bsxfun(@minus, theta, L) / T);
    R = nan(N, Nt+1);
    R(:,end) = (1 - cc(:, end))*CE;
    for t = Nt:-1:1
        R(:,t) = CT*dt(t) + pass_through(:,t) .* R(:,t+1) + (1-pass_through(:,t)).*(1 - cc(:,t))*CE;
    end
    a = R(:,2:end) - (1-cc)*CE;
    cprod_pass_through = cumprod(pass_through, 2);
    semigrad = zeros(size(theta));
    for i=1:length(index)
        semigrad = semigrad + a(i,:) .* cprod_pass_through(i, :) .* (1-pass_through(i,:)) / T;
    end
    semigrad = semigrad / N;
    semicost = mean(R(:,1));
end
end

function visualize_landscape(thetaArray, TArray, L, cc, CE, CT)
[N, Nt] = size(L);
% figure out cost using constant threshold
theta_N = length(thetaArray);
cost = nan(1, theta_N);
for theta_I = 1:theta_N
    [~, ~,~,~,~, ~, cost(theta_I)] = eval_sequential_threshold_cost(thetaArray(theta_I)*ones(1, Nt), TArray, L, cc, CE, CT);
end
[~, m_I] = min(cost);
thetaopt = thetaArray(m_I);

% use a coarser grid
thetaArray = linspace(thetaArray(1), thetaArray(end), 30); theta_N = length(thetaArray);
landscape = nan(theta_N, theta_N);
for theta_I = 1:theta_N
    for theta_J = 1:theta_N
        this_theta = thetaopt*ones(1, Nt);
        this_theta(1) = thetaArray(theta_I);
        this_theta(10) = thetaArray(theta_J);
        [~, ~,~,~,~, ~, landscape(theta_I,theta_J)] = eval_sequential_threshold_cost(this_theta, TArray, L, cc, CE, CT);
    end
end
figure(100); clf;
surf(thetaArray, thetaArray, landscape);

figure(101); clf;
[~, new_m_I] = min(abs(thetaArray-thetaopt));
plot_theta = thetaArray - log(1 + exp(thetaArray));
subplot(121);
plot(plot_theta, landscape(new_m_I, :), 'r--'); hold on;
plot(plot_theta, landscape(:, new_m_I), 'g-'); hold on;
xlabel('theta on log (P/(1-P))'); legend('theta_2', 'theta_1');
subplot(122);
plot(thetaArray, landscape(new_m_I, :), 'r--'); hold on;
plot(thetaArray, landscape(:, new_m_I), 'g-'); hold on;
xlabel('theta on logP');
pause;
end

function [theta, cost, alltheta] = iterative_solve_for_sequential_thresholds(theta_ini, thetaArray, TArray, L, cc, CE, CT)
vec = @(x) reshape(x, numel(x),1);
TArray = vec(TArray)';

theta = theta_ini;
maxIter = 500;
alltheta = cell(1, maxIter);
learning_rate = 0.1;
smoothness = 0.01;
cost = nan(1, maxIter);
batch_size = size(L,1);   %%%%%%%%%%%%% CHANGE ME %%%%%%%%%%%%%%%
batch_N = floor(size(L,1)/batch_size);
T = 0.5;
for iter = 1:maxIter
    
    cost(iter) = 0;
    for batch_I = 1:batch_N
        batch_idx = (batch_I-1)*batch_size + 1 : batch_I*batch_size;
        
        [this_cost, ~,~,~,~, semigrad] = eval_sequential_threshold_cost(theta, TArray, L(batch_idx,:), cc(batch_idx,:), CE, CT, T);
        % add smoother
        [smooth_cost, smoothgrad] = eval_smooth_cost(theta, smoothness);
        semigrad = semigrad + smoothgrad;
        
        theta = theta - semigrad * learning_rate;
        
        cost(iter) = cost(iter) + this_cost;
    end
    cost(iter) = cost(iter)/batch_N;
    if iter>10,
        if cost(iter)>cost(iter-1),
            learning_rate = learning_rate*0.9;
        elseif learning_rate < 10
            learning_rate = learning_rate*1.1;
        end
    end
    T = max(0.01, T * 0.99);
    %learning_rate = learning_rate * 0.999;
    
    alltheta{iter} = theta; % record theta after update
end
%cost(end) = eval_sequential_threshold_cost(theta, TArray, L, cc, CE, CT);

end

function [cost, grad] = eval_smooth_cost(theta, lambda)
aug_theta = [theta(1) theta(:)' theta(end)];
theta_diff = aug_theta(1:end-1) - aug_theta(2:end);
cost = sum(theta_diff.^2)*lambda;
grad = 4*lambda* (aug_theta(2:end-1) -  0.5*(aug_theta(1:end-2)+aug_theta(3:end)));
end


function [theta, R] = solve_for_sequential_thresholds(thetaArray, TArray, L, cc, CE, CT)
vec = @(x) reshape(x, numel(x),1);
TArray = vec(TArray)';
Nt = length(TArray);
dt = [TArray(1) diff(TArray)];
theta = nan(1, Nt);
R = nan(1, Nt);

cost_error = (1-mean(cc, 1))*CE; % [1 x Nt];
R(end) = cost_error(end);
theta(end) = max(thetaArray);
theta_N = length(thetaArray);
for ii = Nt-1:-1:1
    tt = nan(1, theta_N);
    for theta_I = 1:theta_N
        cont = mean(L(:,ii)<thetaArray(theta_I)) ;
        tt(theta_I) = CT*dt(ii) + cont * R(ii+1) + (1-cont)*cost_error(ii);
    end
    [R(ii), theta_mI] = min(tt);
    i_tie = (tt-tt(theta_mI)<1e-10);
    if sum(i_tie)>0, theta_mI = round(median(find(i_tie))); end
    theta(ii) = thetaArray(theta_mI);
end

end


function [theta, R] = solve_for_sequential_thresholds_two_step(thetaArray, TArray, L, cc, CE, CT)
vec = @(x) reshape(x, numel(x),1);
TArray = vec(TArray)';
Nt = length(TArray);
dt = [TArray(1) diff(TArray)];
theta = nan(1, Nt);

cost_error = (1-mean(cc, 1))*CE; % [1 x Nt];
theta(end) = max(thetaArray);
theta_N = length(thetaArray);
R = nan(theta_N, Nt+1);
R(:,end) = cost_error(end);

theta_mI = nan(theta_N, Nt); % store the best theta index for each theta in the previous time-step
% backward pass
for ii = Nt:-1:1
    tt = nan(theta_N, theta_N);
    for theta_prev_I = 1:theta_N
        if ii > 1
            i_good = L(:,ii-1) < thetaArray(theta_prev_I);
        else
            i_good = 1:size(L,1); % the first time slice doesn't have history
        end
        for theta_I = 1:theta_N
            % estimate q by taking both the previous and current
            % thresholds into account
            cont = (sum(L(i_good,ii) < thetaArray(theta_I)) + 1) ./ (length(i_good) + 2 ); % use smoothed estimate
            tt(theta_prev_I, theta_I) = CT*dt(ii) + cont * R(theta_I, ii+1) + (1-cont)*cost_error(ii);
        end
        [R(theta_prev_I, ii), theta_mI(theta_prev_I,ii)] = min(tt(theta_prev_I,:));
    end
end

% recover the solution using DP forward pass
prev_theta_I = 1;
for ii = 1:Nt
    theta(ii) = thetaArray(theta_mI(prev_theta_I, ii));
    prev_theta_I = theta_mI(prev_theta_I, ii);
end
end

