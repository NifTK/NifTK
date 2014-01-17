function [final_params, sumsqs, residuals, exitflag, output, lambda, J] = niftkUltrasoundPinCalibrationOptimisation(initialGuess, tMr_matrices, pin_positions)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Usage:
%   M = niftkUltrasoundPinCalibrationOptimisation(initialGuess, tMr_matrices, pin_positions)
% where:
%   initialGuess : parameters array [tx, ty, tz, rx, ry, rz, x, y, z, s]
%                  where:
%                  tx, ty, tz = translation in millimetres
%                  rx, ry, rz = rotations in radians
%                  s          = isotropic scale factor (mm/pix)
%                  x,y,z      = location of invariant point in millimetres
%
% NOTE: This function should only be called from niftkUltrasoundPinCalibration.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

opt_options = optimset('lsqnonlin') ; % Use a least-square non-linear optimisation
opt_options.LargeScale = 'on';        % Set this as a large scale problem
opt_options.Display = 'iter';         % Display results after each iteration
opt_options.MaxFunEvals = 100000;
opt_options.MaxIter = 100000;

start_params = [initialGuess(1), initialGuess(2), initialGuess(3), initialGuess(4), initialGuess(5), initialGuess(6), initialGuess(7), initialGuess(8), initialGuess(9), initialGuess(10)];
H = ones(size(start_params));

[final_params, sumsqs, residuals, exitflag, output, lambda, J] = lsqnonlin(@CompCalResidual,H.*start_params,[],[],opt_options, tMr_matrices, pin_positions);

