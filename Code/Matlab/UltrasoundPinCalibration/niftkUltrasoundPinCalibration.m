function [finalParams, sumsqs, residuals, iOutliers] = niftkUltrasoundPinCalibration(initialGuess, trackingMatrices, ultrasoundPoints)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Usage:
%   [finalParams, sumsqs, residuals] = niftkUltrasoundPinCalibration(initialGuess, trackingMatrices, ultrasoundPoints)
% where:
%   initialGuess   : parameters array [tx, ty, tz, rx, ry, rz, x, y, z, sx, sy]
%                    where:
%                    tx, ty, tz = translation in millimetres
%                    rx, ry, rz = rotations in radians
%                    x,y,z      = location of invariant point in millimetres
%                    sx, sy     = isotropic scale factor (mm/pix)
%
% trackingMatrices : Cell array of {n, 1} where each cell is a 4x4 tracking matrix.
% ultrasoundPoints : Cell array of {n, 1} where each cell is a 1x4 homogeneous point matrix of [x y 0 1].
%
% This function computes the calibration matrix for the 3DUS system from data collected with a pin phantom.
% The head of the pin forms an invariant point, and is imaged using ultrasound.
%
% The following conventions are used:
%
%   T is the co-ordinate system of the tracking system (Optotrak/Polaris camera)
%   R is the sensor co-ordinate system (IR-LEDs)
%   I is the co-ordinate system of the ultrasound image (top-left corner is the origin)
%   V is the co-ordinate system of the physical world
% 
% aMb is a 4x4 rigid-body transformation matrix from B to A. For example, tMr is the transformation
% from R to T. pS is a position vector of a point in co-ordinate system S (e.g. pI is a point in I).
% If there are N pin images, then for the pin position, (pI)n, on the nth image (1 <= n <= N), 
%
%               pV = vMt.(tMr)n.rMi.(pI)n                                        (1)
%
% where (tMr)n is the transformation matrix for the nth image. vMt and rMi are unknown, but are assumed constant
% for all the images. If we set pV to be at the origin of V, the equation becomes:
%
%               vMt.(tMr)n.rMi.(pI)n = [0 0 0 1]'                                (2)
%
% Using this equation, we can generate a system of 3N (3 rows of the left-hand-side are useful; the 4th just says 1=1)
% non-linear equations and estimate the parameters of vMt and rMi using a least-squares optimization and 
% the Levenberg-Marquardt algorithm. 11 parameters are estimated: the x,y, and z components of the translation vector
% of vMt, and 3 rotation angles and the x,y, and z components of the translation of rMi.
%
% 2014-01-09 Steve Thompson, Matt Clarkson
% 2008-02-25 Dean Barratt
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

iIndex = [];
iOutliers = [];

N = size(trackingMatrices,1);
for i = 1:N;
  iIndex(i) = i;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Here we implement a basic trimmed least squares approach.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%
% Run optimisation
%%%%%%%%%%%%%%%%%%

[finalParams, sumsqs, residuals] = niftkUltrasoundPinCalibrationOptimisation(initialGuess, trackingMatrices, ultrasoundPoints)

disp('Final calibration (image to tracker) matrix:');
rMi = Comp_RigidBody_Matrix(finalParams);
disp(rMi);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Work out the outliers. Only re-run if removing the outliers leaves at least 40 points.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[iOutliers] = niftkUltrasoundPinCalibrationOutliers(finalParams, rMi, trackingMatrices, ultrasoundPoints, iIndex)
M = size(iOutliers,2);
disp('Number of outliers');
disp(M);

if (N-M > 40)

  %%%%%%%%%%%%%%%%%%%%%%
  % Throw away outliers.
  %%%%%%%%%%%%%%%%%%%%%%

  trackingMatrices2 = {};
  ultrasoundPoints2 = {};
  iIndex2 = [];

  counter=1;
  for i = 1:N
    isOutlier = 0;
    for j = 1:M
      if (i == iOutliers(j))
        isOutlier = 1;
      end
    end
    if (isOutlier == 0)
      ultrasoundPoints2{counter,1} = ultrasoundPoints{i};
      trackingMatrices2{counter,1} = trackingMatrices{i};
      iIndex2(counter) = counter;
      counter = counter + 1;
    end
  end

  %%%%%%%%%%%%%%%%%%%%%
  % Re-Run optimisation
  %%%%%%%%%%%%%%%%%%%%%

  disp('Rerunning with this many points:');
  disp(counter);

  [finalParams, sumsqs, residuals] = niftkUltrasoundPinCalibrationOptimisation(initialGuess, trackingMatrices2, ultrasoundPoints2)
  rMi2 = Comp_RigidBody_Matrix(finalParams);

  % Note: Deliberately assigning outliers to different array.
  %       So the output of this whole method, is the COMPLETE list of outliers.
  [iOutliers2] = niftkUltrasoundPinCalibrationOutliers(finalParams, rMi2, trackingMatrices2, ultrasoundPoints2, iIndex2)

  disp('Previous calibration (image to tracker) matrix:');
  disp(rMi);
  disp('Final calibration (image to tracker) matrix:');
  disp(rMi2);
end

disp('Final parameters');
disp(finalParams);




   
   
   
