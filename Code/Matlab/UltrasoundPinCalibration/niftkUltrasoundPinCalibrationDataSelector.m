function [selectedTrackingMatrices, selectedUltrasoundPoints, selectedIndexes] = niftkUltrasoundPinCalibrationDataSelector(allTrackingMatrices, allUltrasoundPoints, numberOfPoints)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Usage:
%   [selectedTrackingMatrices, selectedUltrasoundPoints, selectedIndexes] = niftkUltrasoundPinCalibrationDataSelector(allTrackingMatrices, allUltrasoundPoints, numberOfPoints)
% where:
%
%   TODO: Documentation.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N = size(allTrackingMatrices,1);
selectedIndexes = randperm(N,numberOfPoints);

selectedTrackingMatrices = {};
selectedUltrasoundPoints = {};

for i = 1:numberOfPoints
  selectedTrackingMatrices(i,1) = allTrackingMatrices(selectedIndexes(i),1);
  selectedUltrasoundPoints(i,1) = allUltrasoundPoints(selectedIndexes(i),1);
end





