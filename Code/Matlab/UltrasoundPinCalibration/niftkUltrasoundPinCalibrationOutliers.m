function [outliers] = niftkUltrasoundPinCalibrationOutliers(finalParams, rMi, tMr, pinPositions, iIndex)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Usage:
%   [outliers] = niftkUltrasoundPinCalibrationOutliers(finalParams, rMi, tMr, pinPositions, iIndex)
% where:
%
%   finalParams    : array containing calibration parameters
%   rMi            : 4x4 calibration matrix
%   tMr            : cell array of transformation (tracking) matrices
%   pinPositions   : cell array containing pin positions
%   iIndex         : array of same size as pinPositions, containing 1..n.
%
% NOTE: This function should only be called from niftkUltrasoundPinCalibration.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
outliers = [];
S = diag([finalParams(10) finalParams(10) 1 1]);
ptsR = [];
for i = 1:size(pinPositions,1)
    ptsR = [ptsR tMr{i,1}*rMi*S*pinPositions{i,1}];
end
  
D = ptsR - repmat(median(ptsR,2),1,size(ptsR,2));
D = magc(D(1:3,:));

outliers = iIndex(find(D>3));
plot3d(ptsR,1,'.');
