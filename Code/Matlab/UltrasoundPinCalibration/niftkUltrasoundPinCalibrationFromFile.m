function [finalParams, sumsqs, residuals] = niftkUltrasoundPinCalibrationFromFile(initialGuess)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Usage:
%   [finalParams, sumsqs, residuals] = niftkUltrasoundPinCalibrationFromFile(initialGuess)
% where:
%   initialGuess : parameters array [tx, ty, tz, rx, ry, rz, x, y, z, sx, sy]
%                  where:
%                  tx, ty, tz = translation in millimetres
%                  rx, ry, rz = rotations in radians
%                  x,y,z      = location of invariant point in millimetres
%                  sx, sy     = scale factor (mm/pix)
%
%
% e.g. For the data in NifTKData/Input/UltrasoundPinCalibration/2014.01.07-matt-calib-4DC7
%
% niftkUltrasoundPinCalibrationFromFile [200 -10   0   0   0   20  545 237 -1820 -0.2073 -0.2073]
%
% This functions will ask for:
%
%   matrixFile   : a single file containing N tracking matrices as 4x4 matrices, one after the other.
%   pointFile    : a single file containing N points, with x and y on the same line and seperate
%                  points on each line, and in same order as matrices.
%
% and then calls niftkUltrasoundPinCalibrationFileLoader and niftkUltrasoundPinCalibration.
%
% See additional help there.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ----------------------------------------------------------------------------------------------------------------------
% Load data.
% ----------------------------------------------------------------------------------------------------------------------
[trackingMatrices, ultrasoundPoints, iIndex] = niftkUltrasoundPinCalibrationFileLoader();

% ----------------------------------------------------------------------------------------------------------------------
% Call calibration routine.
% ----------------------------------------------------------------------------------------------------------------------
[finalParams, sumsqs, residuals, outliers] = niftkUltrasoundPinCalibration(initialGuess, trackingMatrices, ultrasoundPoints);

% ----------------------------------------------------------------------------------------------------------------------
% Assess data.
% ----------------------------------------------------------------------------------------------------------------------
iIndex = [];
N = size(trackingMatrices,1);
for i = 1:N;
  iIndex(i) = i;
end
rMi = Comp_RigidBody_Matrix(finalParams);
outliers = niftkUltrasoundPinCalibrationOutliers(finalParams, rMi, trackingMatrices, ultrasoundPoints, iIndex);

disp('Outliers are:');
disp(outliers);


   
