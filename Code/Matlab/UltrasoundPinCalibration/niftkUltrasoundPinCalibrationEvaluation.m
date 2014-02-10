function [meanAccuracy, stdDevAccuracy, meanPrecision, stdDevPrecision] = niftkUltrasoundPinCalibrationEvaluation(initialGuess, goldStandardPoint, numberOfPoints, numberOfRepetitions, sumSquaresThreshold)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Usage:
%   [meanAccuracy, stdDevAccuracy, meanPrecision, stdDevPrecision]
%     = niftkUltrasoundPinCalibrationEvaluation(initialGuess, goldStandardPoint, numberOfPoints, numberOfRepetitions, residualThreshold)
% where:
%   initialGuess : parameters array [tx, ty, tz, rx, ry, rz, x, y, z, sx, sy]
%                  where:
%                  tx, ty, tz = translation in millimetres
%                  rx, ry, rz = rotations in radians
%                  x, y, z    = location of invariant point in millimetres
%                  sx, sy     = scale factor (mm/pix)
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
results = zeros(numberOfRepetitions, 3);
counterForSuccessfulCalibrations = 0;

% ----------------------------------------------------------------------------------------------
% Run one calibration with all data, to get complete list of outliers, relative to median point.
% ----------------------------------------------------------------------------------------------
[finalParams, sumsqs, residuals, outliers] = niftkUltrasoundPinCalibration(initialGuess, allTrackingMatrices, allUltrasoundPoints);

% -------------------------------------------------------------------------
% Now we do evaluation, taking a subset of points, and measuring:
%   accuracy  = RMS error from gold standard
%   precision = RMS error from median reconstructed point
% -------------------------------------------------------------------------
while(true)

  [selectedTrackingMatrices, selectedUltrasoundPoints, selectedIndexes] = niftkUltrasoundPinCalibrationDataSelector(allTrackingMatrices, allUltrasoundPoints, numberOfPoints);
  [finalParams, sumsqs, residuals, outliersFromSelectedData]            = niftkUltrasoundPinCalibration(initialGuess, selectedTrackingMatrices, selectedUltrasoundPoints);

  if (sumsqs < sumSquaresThreshold)

    rMi = Comp_RigidBody_Matrix(finalParams(1:6));
    S = diag([finalParams(10) finalParams(11) 1 1]);

    % -----------------------------------------------------------------------------------
    % Work out median reconstructed point using all data, that is not an obvious outlier.
    % -----------------------------------------------------------------------------------
    counterForMedian = 0;
    reconstructedPoints = [];
    medianReconstructedPoint = zeros(4,1);
    for j = 1:numberOfTrackingMatrices
      if ismember(allIndexes(j), selectedIndexes)
        if ~ismember(allIndexes(j), outliersFromSelectedData)
          if ~ismember(allIndexes(j), outliers)
            reconstructedPoint = allTrackingMatrices{allIndexes(j)}*rMi*S*allUltrasoundPoints{allIndexes(j)};
            reconstructedPoints = [reconstructedPoints reconstructedPoint];
            counterForMedian = counterForMedian + 1;
          end
        end
      end
    end

    if(counterForMedian == 0)
      continue
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
      if ~ismember(allIndexes(j), selectedIndexes)
        if ~ismember(allIndexes(j), outliersFromSelectedData)
          if ~ismember(allIndexes(j), outliers)
            reconstructedPoint = allTrackingMatrices{allIndexes(j)}*rMi*S*allUltrasoundPoints{allIndexes(j)};
            squaredDistanceFromGold = ((reconstructedPoint(1,1) -  gsp(1,1))*(reconstructedPoint(1,1) -  gsp(1,1)) + (reconstructedPoint(2,1) -  gsp(2,1))*(reconstructedPoint(2,1) -  gsp(2,1))+ (reconstructedPoint(3,1) -  gsp(3,1))*(reconstructedPoint(3,1) -  gsp(3,1)));
            squaredDistanceFromMedian = ((reconstructedPoint(1,1) -  medianReconstructedPoint(1,1))*(reconstructedPoint(1,1) -  medianReconstructedPoint(1,1)) + (reconstructedPoint(2,1) -  medianReconstructedPoint(2,1))*(reconstructedPoint(2,1) -  medianReconstructedPoint(2,1))+ (reconstructedPoint(3,1) -  medianReconstructedPoint(3,1))*(reconstructedPoint(3,1) -  medianReconstructedPoint(3,1)));
            rmsErrorFromGold = rmsErrorFromGold + squaredDistanceFromGold;
            rmsErrorFromMedian = rmsErrorFromMedian + squaredDistanceFromMedian;
            counterForStats = counterForStats + 1;
          end
        end
      end
    end
    if (counterForStats == 0)
      continue
    end

    rmsErrorFromGold = rmsErrorFromGold / counterForStats;
    rmsErrorFromMedian = rmsErrorFromMedian / counterForStats;
    rmsErrorFromGold = sqrt(rmsErrorFromGold);
    rmsErrorFromMedian = sqrt(rmsErrorFromMedian);
    counterForSuccessfulCalibrations = counterForSuccessfulCalibrations + 1;
    results(counterForSuccessfulCalibrations, 1) = rmsErrorFromGold;
    results(counterForSuccessfulCalibrations, 2) = rmsErrorFromMedian;
    results(counterForSuccessfulCalibrations, 3) = counterForStats;
    results(counterForSuccessfulCalibrations, 4) = finalParams(1);
    results(counterForSuccessfulCalibrations, 5) = finalParams(2);
    results(counterForSuccessfulCalibrations, 6) = finalParams(3);
    results(counterForSuccessfulCalibrations, 7) = finalParams(4);
    results(counterForSuccessfulCalibrations, 8) = finalParams(5);
    results(counterForSuccessfulCalibrations, 9) = finalParams(6);
    results(counterForSuccessfulCalibrations, 10) = finalParams(7);
    results(counterForSuccessfulCalibrations, 11) = finalParams(8);
    results(counterForSuccessfulCalibrations, 12) = finalParams(9);
    results(counterForSuccessfulCalibrations, 13) = finalParams(10);
    results(counterForSuccessfulCalibrations, 14) = finalParams(11);

  end

  if counterForSuccessfulCalibrations == numberOfRepetitions
    break
  end
end

% ----------------------------------------------------------------------------------------------------------------------
% Finished.
% ----------------------------------------------------------------------------------------------------------------------
means = mean(results,1);
mins = min(results);
maxs = max(results);
stdDevs = std(results, 0, 1);
disp('Number of samples in data');
disp(numberOfTrackingMatrices);
disp('Number of samples in calibration');
disp(numberOfPoints);
disp('Gold standard point');
disp(goldStandardPoint);
disp('Median reconstructed point');
disp(transpose(medianReconstructedPoint));
disp('Mean accuracy');
disp(means(1,1));
disp('Std Dev accuracy');
disp(stdDevs(1,1));
disp('Min (best) accuracy');
disp(mins(1,1));
disp('Max (worse) accuracy');
disp(maxs(1,1));
disp('Mean precision');
disp(means(1,2));
disp('Std Dev precision');
disp(stdDevs(1,2));
disp('Min (best) precision');
disp(mins(1,2));
disp('Max (worse) precision');
disp(maxs(1,2));
disp('Mean points in validation');
disp(means(1,3));
disp('Std Dev points in validation');
disp(stdDevs(1,3));
[bestValue, bestIndex] = min(results(:,1),[],1);
disp('Best calibration index');
disp(bestIndex);
disp('Best calibration value');
disp(bestValue);
disp('Best calibration');
bestCalib = results(bestIndex,4:14);
disp(bestCalib);
rMi = Comp_RigidBody_Matrix(bestCalib(1:6));
S = diag([bestCalib(10) bestCalib(11) 1 1]);
disp('Best calibration Matrix');
disp(rMi);
disp('Best calibration Scaling Matrix');
disp(S);
