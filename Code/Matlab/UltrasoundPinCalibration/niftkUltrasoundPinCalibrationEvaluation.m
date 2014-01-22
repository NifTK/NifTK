function [meanAccuracy, stdDevAccuracy, meanPrecision, stdDevPrecision] = niftkUltrasoundPinCalibrationEvaluation(initialGuess, goldStandardPoint, numberOfPoints, numberOfRepetitions, sumSquaresThreshold)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Usage:
%   [meanAccuracy, stdDevAccuracy, meanPrecision, stdDevPrecision]
%     = niftkUltrasoundPinCalibrationEvaluation(initialGuess, goldStandardPoint, numberOfPoints, numberOfRepetitions, residualThreshold)
% where:
%   initialGuess : parameters array [tx, ty, tz, rx, ry, rz, x, y, z, s]
%                  where:
%                  tx, ty, tz = translation in millimetres
%                  rx, ry, rz = rotations in radians
%                  s          = isotropic scale factor (mm/pix)
%                  x,y,z      = location of invariant point in millimetres
%
%   goldStandardPoint   : [x y z] = location of invariant point in millimetres.
%   numberOfPoints      : the number of randomly selected points to use to calibrate.
%   numberOfRepetitions : the number of repetitions of the whole calibration, to build up statistics of mean (stdDev).
%   sumSquaresThreshold : a threshold, above which the calibration is deemed to be obviously garbage and hence rejected.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ----------------------------------------------------------------------------------------------------------------------
% Load all data.
% ----------------------------------------------------------------------------------------------------------------------
[allTrackingMatrices, allUltrasoundPoints, allIndexes] = niftkUltrasoundPinCalibrationFileLoader();
numberOfTrackingMatrices = size(allTrackingMatrices,1);

disp('Total number of samples');
disp(numberOfTrackingMatrices);

gsp = transpose(goldStandardPoint);
disp('Gold standard');
disp(goldStandardPoint);

% ----------------------------------------------------------------------------------------------------------------------
% Basic plan:
%   1. Randomly select `numberOfPoints` points and matrices.
%   2. Calibrate
%   3. Use remaining points to assess how close we are to gold standard point = accuracy.
%   4. Use remaining points to assess how close we are to centre of reconstructed points = precision.
%   5. Repeat `numberOfRepetitions` times to get statistics for mean and standard deviation.
% ----------------------------------------------------------------------------------------------------------------------
results = zeros(numberOfRepetitions, 2);
counterForSuccessfulCalibrations = 0;

% --------------------------------------------------------------------
% Run one calibration with all data, to get complete list of outliers.
% --------------------------------------------------------------------
[finalParams, sumsqs, residuals, outliers] = niftkUltrasoundPinCalibration(initialGuess, allTrackingMatrices, allUltrasoundPoints);

% -------------------------------------------------------------------------
% Now we do evaluation, taking a subset of points, and measuring:
%   accuracy  = RMS error from gold standard
%   precision = RMS error from mean reconstructed point
% -------------------------------------------------------------------------
while(true)

  [selectedTrackingMatrices, selectedUltrasoundPoints, selectedIndexes] = niftkUltrasoundPinCalibrationDataSelector(allTrackingMatrices, allUltrasoundPoints, numberOfPoints);
  [finalParams, sumsqs, residuals, outliersFromSelectedData]            = niftkUltrasoundPinCalibration(initialGuess, selectedTrackingMatrices, selectedUltrasoundPoints);

  if (sumsqs < sumSquaresThreshold)

    rMi = Comp_RigidBody_Matrix(finalParams(1:6));
    S = diag([finalParams(10) finalParams(10) 1 1]);

    % -------------------------------------------------------------------------
    % Work out mean reconstructed point using all data, that is not an outlier.
    % -------------------------------------------------------------------------
    counterForMean = 0;
    meanReconstructedPoint = zeros(4,1);
    for j = 1:numberOfTrackingMatrices
      if ismember(allIndexes(j), selectedIndexes)
        if ~ismember(allIndexes(j), outliersFromSelectedData)
          reconstructedPoint = allTrackingMatrices{allIndexes(j)}*rMi*S*allUltrasoundPoints{allIndexes(j)};
          meanReconstructedPoint = plus(meanReconstructedPoint, reconstructedPoint);
          counterForMean = counterForMean + 1;
        end
      end
    end
    if(counterForMean == 0)
      continue
    end

    meanReconstructedPoint = times(meanReconstructedPoint,1.0/counterForMean);

    % ---------------------------------------------------------------
    % Work out stats relative to mean point, and gold standard point.
    % ---------------------------------------------------------------
    counterForStats = 0;
    squaredDistanceFromGold = 0;
    squaredDistanceFromMean = 0;
    rmsErrorFromGold = 0;
    rmsErrorFromMean = 0;

    % Calculate RMS error.
    for j = 1:numberOfTrackingMatrices
      if ~ismember(allIndexes(j), selectedIndexes)
        if ~ismember(allIndexes(j), outliersFromSelectedData)
          if ~ismember(allIndexes(j), outliers)
            reconstructedPoint = allTrackingMatrices{allIndexes(j)}*rMi*S*allUltrasoundPoints{allIndexes(j)};
            squaredDistanceFromGold = ((reconstructedPoint(1,1) -  gsp(1,1))*(reconstructedPoint(1,1) -  gsp(1,1)) + (reconstructedPoint(2,1) -  gsp(2,1))*(reconstructedPoint(2,1) -  gsp(2,1))+ (reconstructedPoint(3,1) -  gsp(3,1))*(reconstructedPoint(3,1) -  gsp(3,1)));
            squaredDistanceFromMean = ((reconstructedPoint(1,1) -  meanReconstructedPoint(1,1))*(reconstructedPoint(1,1) -  meanReconstructedPoint(1,1)) + (reconstructedPoint(2,1) -  meanReconstructedPoint(2,1))*(reconstructedPoint(2,1) -  meanReconstructedPoint(2,1))+ (reconstructedPoint(3,1) -  meanReconstructedPoint(3,1))*(reconstructedPoint(3,1) -  meanReconstructedPoint(3,1)));
            rmsErrorFromGold = rmsErrorFromGold + squaredDistanceFromGold;
            rmsErrorFromMean = rmsErrorFromMean + squaredDistanceFromMean;
            counterForStats = counterForStats + 1;
          end
        end
      end
    end
    if (counterForStats == 0)
      continue
    end

    rmsErrorFromGold = rmsErrorFromGold / counterForStats;
    rmsErrorFromMean = rmsErrorFromMean / counterForStats;
    rmsErrorFromGold = sqrt(rmsErrorFromGold);
    rmsErrorFromMean = sqrt(rmsErrorFromMean);
    counterForSuccessfulCalibrations = counterForSuccessfulCalibrations + 1;
    results(counterForSuccessfulCalibrations, 1) = rmsErrorFromGold;
    results(counterForSuccessfulCalibrations, 2) = rmsErrorFromMean;

  end

  if counterForSuccessfulCalibrations == numberOfRepetitions
    break
  end
end

% ----------------------------------------------------------------------------------------------------------------------
% Finished.
% ----------------------------------------------------------------------------------------------------------------------
means = mean(results,1);
stdDevs = std(results, 0, 1);
disp('Number of samples in data');
disp(numberOfTrackingMatrices);
disp('Number of samples in calibration');
disp(numberOfPoints);
disp('Number of samples in validation');
disp(numberOfTrackingMatrices - numberOfPoints);
disp('Gold standard point');
disp(goldStandardPoint);
disp('Mean reconstructed point');
disp(transpose(meanReconstructedPoint));
disp('Mean accuracy');
disp(means(1,1));
disp('Std Dev accuracy');
disp(stdDevs(1,1));
disp('Mean precision');
disp(means(1,2));
disp('Std Dev precision');
disp(stdDevs(1,2));

   
