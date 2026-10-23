function run_reference_comparison()
% Generate shared MATLAB fNIRS inputs and reference first-level GLM results.

output_dir = getenv('PYBRAIN_MATLAB_COMPARISON_OUTPUT');
iterations = str2double(getenv('PYBRAIN_MATLAB_COMPARISON_ITERATIONS'));
seed = str2double(getenv('PYBRAIN_MATLAB_COMPARISON_SEED'));
simulation = getenv('PYBRAIN_MATLAB_COMPARISON_SIMULATION');
if isempty(simulation)
    simulation = 'simData';
end
if isempty(output_dir) || isnan(iterations) || iterations < 1 || ...
        isnan(seed) || iterations ~= floor(iterations)
    error('Invalid MATLAB comparison configuration in environment.');
end

input_dir = fullfile(output_dir, 'inputs');
if ~exist(input_dir, 'dir')
    mkdir(input_dir);
end

rng(seed, 'twister');
all_statistics = table();
complex_intensity_iterations = zeros(0, 1);
for iteration = 1:iterations
    switch simulation
        case 'simData'
            [raw_data, truth] = nirs.testing.simData();
        case 'simMotionArtifact'
            [raw_data, truth] = nirs.testing.simMotionArtifact([], 2, 0.5);
        otherwise
            error('Unsupported comparison simulation: %s', simulation);
    end
    if ~isreal(raw_data.data)
        warning('pyBrainAnalyzIR:ComplexSimulatedIntensity', ...
            ['The MATLAB simulation produced complex-valued intensities in ' ...
            'iteration %d; using their real component for the comparison.'], ...
            iteration);
        raw_data.data = real(raw_data.data);
        complex_intensity_iterations(end + 1, 1) = iteration; %#ok<AGROW>
    end
    if any(~isfinite(raw_data.data(:))) || any(raw_data.data(:) <= 0)
        error('Simulated raw intensities must be finite and strictly positive.');
    end
    raw_links = raw_data.probe.link;
    nirs.io.saveSNIRF(raw_data, char(fullfile(input_dir, ...
        sprintf('iteration_%03d', iteration))));

    pipeline = nirs.modules.OpticalDensity();
    pipeline = nirs.modules.BeerLambertLaw(pipeline);
    pipeline.PPF = 6;
    pipeline = nirs.modules.Resample(pipeline);
    pipeline.Fs = 4;
    pipeline = nirs.modules.GLM(pipeline);
    pipeline.type = 'AR-IRLS';
    pipeline.options.Pmax = 16;
    pipeline.verbose = false;

    stats = pipeline.run(raw_data);
    stats = stats(1);
    variables = stats.variables;
    source = variables.source;
    detector = variables.detector;
    signal_type = string(variables.type);
    condition = string(variables.cond);
    beta = stats.beta(:);
    t_value = stats.tstat(:);
    p_value = stats.p(:);
    dof = repmat(stats.dfe(1), height(variables), 1);

    if numel(beta) ~= height(variables) || ...
            numel(t_value) ~= height(variables) || ...
            numel(p_value) ~= height(variables) || ...
            numel(dof) ~= height(variables)
        error('MATLAB GLM estimates do not align with the statistics table.');
    end

    truth_by_estimate = nan(height(variables), 1);
    for row = 1:height(variables)
        link_rows = raw_links.source == source(row) & ...
            raw_links.detector == detector(row);
        link_truth = truth(link_rows);
        if any(isfinite(link_truth))
            truth_by_estimate(row) = double(any(link_truth == 1));
        end
    end

    iteration_column = repmat(iteration, height(variables), 1);
    current_statistics = table(iteration_column, source, detector, ...
        signal_type, condition, beta, t_value, p_value, dof, truth_by_estimate, ...
        'VariableNames', {'iteration', 'source', 'detector', 'signal_type', ...
        'condition', 'beta', 't_value', 'p_value', 'dof', 'truth'});
    all_statistics = [all_statistics; current_statistics]; %#ok<AGROW>
end

statistics_path = fullfile(output_dir, 'matlab_statistics.csv');
writetable(all_statistics, statistics_path);

all_roc = table();
signal_types = unique(all_statistics.signal_type);
for signal_index = 1:numel(signal_types)
    signal_type = signal_types(signal_index);
    selected = all_statistics.signal_type == signal_type & ...
        all_statistics.condition == "A" & ...
        isfinite(all_statistics.truth) & isfinite(all_statistics.p_value);
    [true_positive_rate, false_positive_rate, threshold] = ...
        nirs.testing.roc(all_statistics.truth(selected), ...
        all_statistics.p_value(selected));
    point = (1:numel(threshold))';
    current_roc = table(repmat(signal_type, numel(point), 1), point, ...
        true_positive_rate, false_positive_rate, threshold, ...
        'VariableNames', {'signal_type', 'point', 'true_positive_rate', ...
        'false_positive_rate', 'threshold'});
    all_roc = [all_roc; current_roc]; %#ok<AGROW>
end
writetable(all_roc, fullfile(output_dir, 'matlab_roc.csv'));

metadata = struct( ...
    'matlab_version', version, ...
    'simulation', simulation, ...
    'simulation_function', which(['nirs.testing.' simulation]), ...
    'iterations', iterations, ...
    'seed', seed, ...
    'matlab_ppf', 6, ...
    'sampling_frequency_hz', 4, ...
    'glm_type', 'AR-IRLS', ...
    'ar_order', 16, ...
    'hrf_basis', 'nirs.design.basis.Canonical', ...
    'complex_intensity_real_component_iterations', ...
    complex_intensity_iterations);
if strcmp(simulation, 'simMotionArtifact')
    metadata.motion_spikes_per_minute = 2;
    metadata.motion_shifts_per_minute = 0.5;
end
fid = fopen(fullfile(output_dir, 'matlab_metadata.json'), 'w');
if fid < 0
    error('Could not create MATLAB comparison metadata.');
end
fprintf(fid, '%s', jsonencode(metadata));
fclose(fid);
end
