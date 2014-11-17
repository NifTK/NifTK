function [neural_density, orientation_dispersion_index, csf_volume_fraction, objective_function, kappa_concentration, error, fibre_orientations] = noddi_fitting(dwis, mask, bvals, bvecs, fname)

disp('******************************');
disp('   NODDI fitting pipeline     ');
disp('******************************');

[~, ~, dwis_ext] = fileparts(dwis);
[~, ~, mask_ext] = fileparts(mask);

if (strcmp(dwis_ext,'.gz'))
    disp('nifti gzip detected. gunzip DWIs');
    dwis = gunzip(dwis, pwd);
    dwis = dwis{1};
end
if (strcmp(mask_ext,'.gz'))
    disp('nifti gzip detected. gunzip mask');
    mask = gunzip(mask, pwd);
    mask = mask{1};
end

roi = 'NODDI_roi.mat';

disp('Creating ROI...');
CreateROI(dwis, mask, roi)

disp('Creating Protocol...');
protocol = FSL2Protocol(bvals, bvecs);

disp('Creating Model...');
noddi = MakeModel('WatsonSHStickTortIsoV_B0');

output_params = 'FittedParams.mat';

disp('Fitting NODDI model...');
batch_fitting_single(roi, protocol, noddi, output_params);

disp('Saving outputs...');
SaveParamsAsNIfTI(output_params, roi, mask, fname)

neural_density = strcat(fname, '_ficvf.nii');
orientation_dispersion_index = strcat(fname, '_odi.nii');
csf_volume_fraction = strcat(fname, '_fiso.nii');
objective_function = strcat(fname, '_fmin.nii');
kappa_concentration = strcat(fname, '_kappa.nii');
error = strcat(fname, '_error_code.nii');

fibre_orientations{1} = strcat(fname, '_fibredirs_xvec.nii');
fibre_orientations{2} = strcat(fname, '_fibredirs_yvec.nii');
fibre_orientations{3} = strcat(fname, '_fibredirs_zvec.nii');

disp('******************************');
disp(' NODDI fitting pipeline: done');
disp('******************************');
