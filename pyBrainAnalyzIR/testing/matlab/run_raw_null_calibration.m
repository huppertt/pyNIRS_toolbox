function run_raw_null_calibration()
% Fit the exact Python-generated null recordings; do not generate MATLAB noise.
input_path = getenv('PYBRAIN_RAW_NULL_INPUT');
output_path = getenv('PYBRAIN_RAW_NULL_OUTPUT');
if isempty(input_path) || isempty(output_path) || ~exist(input_path, 'file')
    error('Shared raw-null input and output paths must be configured.');
end
inputs = load(input_path);
[~, n_channels, n_motion, n_replicates] = size(inputs.data);
beta = nan(n_channels, n_motion, n_replicates);
variance = beta;
module_variance = beta;
internal_p = beta;
ar_order = beta;
filter_dc = beta;
dfe = nan(n_motion, n_replicates);
module_indefinite = false(n_motion, n_replicates);
for replicate = 1:n_replicates
    for motion = 1:n_motion
        stats = nirs.math.ar_irls(inputs.data(:, :, motion, replicate), ...
            inputs.design, double(inputs.pmax), [], false, false, false);
        beta(:, motion, replicate) = stats.beta(2, :)';
        internal_p(:, motion, replicate) = stats.pval(2, :)';
        ar_order(:, motion, replicate) = stats.P;
        filter_dc(:, motion, replicate) = cellfun(@sum, stats.filter);
        dfe(motion, replicate) = stats.dfe;
        task_cov = zeros(2 * n_channels);
        for left = 1:n_channels
            variance(left, motion, replicate) = stats.covb(2, 2, left, left);
            for right = 1:n_channels
                task_cov(left + (0:1) * n_channels, right + (0:1) * n_channels) = ...
                    stats.covb(2:3, 2:3, left, right);
            end
        end
        % Exact final covariance transformation in nirs.modules.AR_IRLS.
        module_indefinite(motion, replicate) = min(eig((task_cov + task_cov') / 2)) < 0;
        [u, s, v] = svd(task_cov);
        task_cov = .5 * (u * s * u' + v * s * v');
        module_variance(:, motion, replicate) = diag(task_cov(1:n_channels, 1:n_channels));
    end
    if mod(replicate, 20) == 0
        fprintf('\n%d/%d matched MATLAB recordings\n', replicate, n_replicates);
    end
end
ar_irls_path = which('nirs.math.ar_irls');
ar_fit_path = which('nirs.math.ar_fit');
save(output_path, 'beta', 'variance', 'module_variance', 'internal_p', ...
    'ar_order', 'filter_dc', 'dfe', 'module_indefinite', 'ar_irls_path', 'ar_fit_path');
end
