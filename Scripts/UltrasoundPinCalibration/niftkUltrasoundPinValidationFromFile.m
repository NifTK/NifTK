function niftkUltrasoundPinValidationFromFile(calibrationParams, goldStandardPoint)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Usage:
%   niftkUltrasoundPinValidationFromFile(calibrationParams, goldStandardPoint)
% where:
%   initialGuess : parameters array [tx, ty, tz, rx, ry, rz, x, y, z, sx, sy]
%                  where:
%                  tx, ty, tz = translation in millimetres
%                  rx, ry, rz = rotations in radians
%                  x,y,z      = location of invariant point in millimetres
%                  sx, sy     = scale factor (mm/pix)
%
%   goldStandardPoint = [x, y, z] in millimetres
%
% This functions will ask for:
%
%   matrixFile   : a single file containing N tracking matrices as 4x4 matrices, one after the other.
%   pointFile    : a single file containing N points, with x and y on the same line and seperate
%                  points on each line, and in same order as matrices.
%
% and then computes the distance of the reconstructed points from the gold standard point.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ----------------------------------------------------------------------------------------------------------------------
% Load data.
% ----------------------------------------------------------------------------------------------------------------------
[trackingMatrices, ultrasoundPoints, iIndex] = niftkUltrasoundPinCalibrationFileLoader();
numberOfTrackingMatrices = size(trackingMatrices,1);

gsp = transpose(goldStandardPoint);

results = zeros(numberOfTrackingMatrices, 3);

% ----------------------------------------------------------------------------------------------------------------------
% Construct a single calibration using the given parameters.
% ----------------------------------------------------------------------------------------------------------------------
rMi = Comp_RigidBody_Matrix(calibrationParams(1:6));
S = diag([calibrationParams(10) calibrationParams(11) 1 1]);


% ----------------------------------------------------------------------------------------------------------------------
% Calculate some stats.
% ----------------------------------------------------------------------------------------------------------------------
counterForMedian = 0;
reconstructedPoints = [];
medianReconstructedPoint = zeros(4,1);
for j = 1:numberOfTrackingMatrices
  reconstructedPoint = trackingMatrices{j}*rMi*S*ultrasoundPoints{j};
  reconstructedPoints = [reconstructedPoints reconstructedPoint];
  counterForMedian = counterForMedian + 1;
end
medianReconstructedPoint = median(reconstructedPoints, 2);

% -----------------------------------------------------------------
% Work out stats relative to median point, and gold standard point.
% -----------------------------------------------------------------
counterForStats = 0;
squaredDistanceFromGold = 0;
squaredDistanceFromMedian = 0;
rmsErrorFromGold = 0;
rmsErrorFromMedian = 0;

% Calculate RMS error.
for j = 1:numberOfTrackingMatrices
  reconstructedPoint = trackingMatrices{j}*rMi*S*ultrasoundPoints{j};
  squaredDistanceFromGold = ((reconstructedPoint(1,1) -  gsp(1,1))*(reconstructedPoint(1,1) -  gsp(1,1)) + (reconstructedPoint(2,1) -  gsp(2,1))*(reconstructedPoint(2,1) -  gsp(2,1))+ (reconstructedPoint(3,1) -  gsp(3,1))*(reconstructedPoint(3,1) -  gsp(3,1)));
  squaredDistanceFromMedian = ((reconstructedPoint(1,1) -  medianReconstructedPoint(1,1))*(reconstructedPoint(1,1) -  medianReconstructedPoint(1,1)) + (reconstructedPoint(2,1) -  medianReconstructedPoint(2,1))*(reconstructedPoint(2,1) -  medianReconstructedPoint(2,1))+ (reconstructedPoint(3,1) -  medianReconstructedPoint(3,1))*(reconstructedPoint(3,1) -  medianReconstructedPoint(3,1)));
  rmsErrorFromGold = rmsErrorFromGold + squaredDistanceFromGold;
  rmsErrorFromMedian = rmsErrorFromMedian + squaredDistanceFromMedian;
  counterForStats = counterForStats + 1;
end

rmsErrorFromGold = rmsErrorFromGold / counterForStats;
rmsErrorFromMedian = rmsErrorFromMedian / counterForStats;
rmsErrorFromGold = sqrt(rmsErrorFromGold);
rmsErrorFromMedian = sqrt(rmsErrorFromMedian);

disp('Gold standard point');
disp(goldStandardPoint);
disp('Median reconstructed point');
disp(transpose(medianReconstructedPoint));
disp('RMS from gold');
disp(rmsErrorFromGold);
disp('RMS from median');
disp(rmsErrorFromMedian);
