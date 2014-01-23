function [finalParams, sumsqs, residuals] = niftkUltrasoundPinCalibrationFromFile(initialGuess)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Usage:
%   [finalParams, sumsqs, residuals] = niftkUltrasoundPinCalibrationFromFile(initialGuess)
% where:
%   initialGuess : parameters array [tx, ty, tz, rx, ry, rz, x, y, z, s]
%                  where:
%                  tx, ty, tz = translation in millimetres
%                  rx, ry, rz = rotations in radians
%                  s          = isotropic scale factor (mm/pix)
%                  x,y,z      = location of invariant point in millimetres
%
%
% e.g. For the data in NifTKData/Input/UltrasoundPinCalibration/2014.01.07-matt-calib-4DC7
%
% niftkUltrasoundPinCalibrationFromFile [200 -10   0   0   0   20  545 237 -1820 -0.2073]
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
[finalParams, sumsqs, residuals] = niftkUltrasoundPinCalibration(initialGuess, trackingMatrices, ultrasoundPoints);
   
   
