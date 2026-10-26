function run_roi_glm_comparison()
% Run shared synthetic time courses through both GLM/ROI processing orders.
input_path = getenv('PYBRAIN_ROI_GLM_INPUT');
output_path = getenv('PYBRAIN_ROI_GLM_OUTPUT');
if isempty(input_path) || isempty(output_path) || ~exist(input_path, 'file')
    error('ROI/GLM comparison input and output paths must be configured.');
end

inputs = load(input_path);
[n_time, n_channels, n_examples] = size(inputs.data);
n_rois = size(inputs.roi_weights, 1);
max_conditions = 2;
beta_timecourse_first = nan(n_rois, max_conditions, n_examples);
covariance_timecourse_first = nan(n_rois * max_conditions, ...
    n_rois * max_conditions, n_examples);
beta_glm_first = nan(n_rois, max_conditions, n_examples);
covariance_glm_first = nan(n_rois * max_conditions, ...
    n_rois * max_conditions, n_examples);

for iteration = 1:n_examples
    n_conditions = double(inputs.num_conditions(iteration));
    n_coefficients = n_conditions + 1;
    design = inputs.design(:, 1:n_coefficients, iteration);
    signal = inputs.data(:, :, iteration);
    weights = inputs.roi_weights(:, :, iteration);
    if rank(design) < n_coefficients || any(abs(sum(weights, 2) - 1) > 1e-12)
        error('Each design must be full rank and ROI weights must sum to one.');
    end
    covariance_design = (design' * design) \ eye(n_coefficients);
    task_columns = 2:n_coefficients;
    output_indices = [];

    roi_signal = signal * weights';
    roi_coefficients = design \ roi_signal;
    roi_residuals = roi_signal - design * roi_coefficients;
    roi_residual_covariance = (roi_residuals' * roi_residuals) / ...
        (n_time - n_coefficients);
    roi_full_covariance = kron(roi_residual_covariance, covariance_design);
    output_indices = [];
    padded_indices = [];
    for roi = 1:n_rois
        beta_timecourse_first(roi, 1:n_conditions, iteration) = ...
            roi_coefficients(task_columns, roi)';
        output_indices = [output_indices, ...
            (roi - 1) * n_coefficients + task_columns]; %#ok<AGROW>
        padded_indices = [padded_indices, ...
            (roi - 1) * max_conditions + (1:n_conditions)]; %#ok<AGROW>
    end
    covariance_timecourse_first(padded_indices, padded_indices, iteration) = ...
        roi_full_covariance(output_indices, output_indices);

    channel_coefficients = design \ signal;
    residuals = signal - design * channel_coefficients;
    channel_residual_covariance = (residuals' * residuals) / ...
        (n_time - n_coefficients);
    channel_full_covariance = kron(channel_residual_covariance, covariance_design);
    projection = zeros(n_rois * n_coefficients, n_channels * n_coefficients);
    for roi = 1:n_rois
        for channel = 1:n_channels
            roi_rows = (roi - 1) * n_coefficients + (1:n_coefficients);
            channel_columns = (channel - 1) * n_coefficients + (1:n_coefficients);
            projection(roi_rows, channel_columns) = ...
                weights(roi, channel) * eye(n_coefficients);
        end
    end
    pooled_coefficients = projection * channel_coefficients(:);
    pooled_covariance = projection * channel_full_covariance * projection';
    output_indices = [];
    for roi = 1:n_rois
        beta_glm_first(roi, 1:n_conditions, iteration) = ...
            pooled_coefficients((roi - 1) * n_coefficients + task_columns)';
        output_indices = [output_indices, ...
            (roi - 1) * n_coefficients + task_columns]; %#ok<AGROW>
    end
    padded_indices = [];
    for roi = 1:n_rois
        padded_indices = [padded_indices, ...
            (roi - 1) * max_conditions + (1:n_conditions)]; %#ok<AGROW>
    end
    covariance_glm_first(padded_indices, padded_indices, iteration) = ...
        pooled_covariance(output_indices, output_indices);
end

save(output_path, 'beta_timecourse_first', 'covariance_timecourse_first', ...
    'beta_glm_first', 'covariance_glm_first');
end
