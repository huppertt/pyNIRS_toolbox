function run_roi_ar_irls_comparison()
% Run shared synthetic time courses through nirs-toolbox AR-IRLS with
% inverse-variance ROI averaging, in both processing orders.
input_path = getenv('PYBRAIN_ROI_GLM_INPUT');
output_path = getenv('PYBRAIN_ROI_GLM_OUTPUT');
if isempty(input_path) || isempty(output_path) || ~exist(input_path, 'file')
    error('ROI/GLM comparison input and output paths must be configured.');
end

inputs = load(input_path);
p_max = double(inputs.ar_order);
[~, n_channels, n_examples] = size(inputs.data);
n_rois = size(inputs.roi_weights, 1);
max_conditions = 2;
beta_timecourse_first = nan(n_rois, max_conditions, n_examples);
covariance_timecourse_first = nan(n_rois * max_conditions, ...
    n_rois * max_conditions, n_examples);
beta_glm_first = nan(n_rois, max_conditions, n_examples);
covariance_glm_first = nan(n_rois * max_conditions, ...
    n_rois * max_conditions, n_examples);

link = table((1:n_channels)', (1:n_channels)', repmat({'hbo'}, n_channels, 1), ...
    'VariableNames', {'source', 'detector', 'type'});
probe = nirs.core.Probe([(1:n_channels)' zeros(n_channels, 2)], ...
    [(1:n_channels)' + 0.5, ones(n_channels, 1), zeros(n_channels, 1)], link);
roi_names = arrayfun(@(k) sprintf('ROI_%d', k), 1:n_rois, 'UniformOutput', false);

for iteration = 1:n_examples
    n_conditions = double(inputs.num_conditions(iteration));
    design = inputs.design(:, 1:n_conditions + 1, iteration);
    signal = inputs.data(:, :, iteration);
    weights = inputs.roi_weights(:, :, iteration);
    condition_names = arrayfun(@(k) sprintf('task_%d', k), 1:n_conditions, ...
        'UniformOutput', false);

    % ROI time course first: inverse temporal variance weights per channel.
    effective = weights ./ var(signal, 1, 1);
    effective = effective ./ sum(effective, 2);
    roi_signal = signal * effective';
    [beta, covb] = fit_ar_irls(roi_signal, design, p_max);
    beta_timecourse_first(:, 1:n_conditions, iteration) = beta';
    padded = padded_indices(n_rois, n_conditions, max_conditions);
    covariance_timecourse_first(padded, padded, iteration) = ...
        reorder_roi_covariance(covb, n_rois, n_conditions);

    % GLM first, then nirs.util.roiAverage on the ChannelStats (weighted).
    [beta, covb] = fit_ar_irls(signal, design, p_max);
    stats = nirs.core.ChannelStats;
    stats.variables = [repmat(link, [n_conditions 1]), ...
        table(reshape(repmat(condition_names, n_channels, 1), [], 1), ...
        'VariableNames', {'cond'})];
    stats.beta = reshape(beta', [], 1);
    stats.covb = covb;
    stats.dfe = size(signal, 1) - size(design, 2);
    stats.probe = probe;
    roi_tables = cell(1, n_rois);
    for roi = 1:n_rois
        members = find(weights(roi, :) > 0)';
        roi_tables{roi} = table(members, members, weights(roi, members)', ...
            repmat(roi_names(roi), numel(members), 1), ...
            'VariableNames', {'source', 'detector', 'weight', 'Name'});
    end
    [~, roi_stats] = nirs.util.roiAverage(stats, roi_tables, roi_names, false, true);
    pooled_beta = reshape(roi_stats.beta, n_rois, n_conditions);
    pooled_covb = reorder_roi_covariance(roi_stats.covb, n_rois, n_conditions);
    beta_glm_first(:, 1:n_conditions, iteration) = pooled_beta;
    padded = padded_indices(n_rois, n_conditions, max_conditions);
    covariance_glm_first(padded, padded, iteration) = pooled_covb;
end

save(output_path, 'beta_timecourse_first', 'covariance_timecourse_first', ...
    'beta_glm_first', 'covariance_glm_first');
end

function [beta, covb] = fit_ar_irls(signal, design, p_max)
% nirs.math.ar_irls with an explicit design; return task coefficients and
% the condition-major covariance used by nirs.modules.AR_IRLS.
stats = nirs.math.ar_irls(signal, design, p_max, [], false);
n_channels = size(signal, 2);
n_tasks = size(design, 2) - 1;
beta = stats.beta(2:end, :);
covb = zeros(n_channels * n_tasks);
for j = 1:n_channels
    for k = 1:n_channels
        rows = (0:n_tasks - 1) * n_channels + j;
        columns = (0:n_tasks - 1) * n_channels + k;
        covb(rows, columns) = stats.covb(2:end, 2:end, j, k);
    end
end
end

function indices = padded_indices(n_rois, n_conditions, max_conditions)
indices = [];
for roi = 1:n_rois
    indices = [indices, (roi - 1) * max_conditions + (1:n_conditions)]; %#ok<AGROW>
end
end

function covb = reorder_roi_covariance(covb, n_rois, n_conditions)
% Condition-major (condition, roi) -> roi-major (roi, condition).
order = [];
for roi = 1:n_rois
    order = [order, (0:n_conditions - 1) * n_rois + roi]; %#ok<AGROW>
end
covb = covb(order, order);
end
