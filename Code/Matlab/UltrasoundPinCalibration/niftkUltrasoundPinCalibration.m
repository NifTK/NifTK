function M = niftkUltrasoundPinCalibration(initialGuess)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Usage:
%   M = niftkUltrasoundPinCalibration(initialGuess)
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
% niftkUltrasoundPinCalibration [200 -10   0   0   0   20  545 237 -1820 -0.2073]
%
% This functions will ask for:
%
%   matrixFile   : a single file containing N tracking matrices as 4x4 matrices, one after the other.
%   pointFile    : a single file containing N points, with x and y on the same line and seperate
%                  points on each line, and in same order as matrices.
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
% the Levenberg-Marquardt algorithm. 10 parameters are estimated: the x,y, and z components of the translation vector
% of vMt, and 3 rotation angles and the x,y, and z components of the translation of rMi.
%
% 2014-01-09 Steve Thompson, Matt Clarkson
% 2008-02-25 Dean Barratt
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

file_selection = {'*.txt','Matrix files (*.txt)'};
[matrixFilename, matrixPathname] = uigetfile(file_selection,'Select .txt file containing all matrices');
if matrixFilename == 0  % Return if Cancel is pressed
   return;
end

file_selection = {'*.txt','Point files (*.txt)'};
[pointFilename, pointPathname] = uigetfile(file_selection,'Select .txt file containing all points');
if pointFilename == 0  % Return if Cancel is pressed
   return;
end

file_id = fopen(matrixFilename, 'r');
[T,countT] = fscanf(file_id, '%f');
fclose(file_id);
fprintf('\n %d matrices read in...\n', countT/16);

file_id = fopen(pointFilename, 'r');
[P,countP] = fscanf(file_id, '%f');
fclose(file_id);
fprintf('\n %d points read in...\n', countP/2);

if (countP/2 ~= countT/16)
  errordlg('Number of points does not equal number of matrices!');
  return
end

N = countP/2;

% ----------------------------------------------------------------------------------------------------------------------
% Setup matrices.
% ----------------------------------------------------------------------------------------------------------------------

i_outliers = [];
tMr_matrices = {}; pin_positions = {}; i_index = [];
for i = 1:N
  pin_positions{i,1} = transpose([P(i*2-1), P(i*2), 0, 1]);
  tMr_matrices{i,1}  = [[T(i*16-15),T(i*16-14),T(i*16-13),T(i*16-12)];[T(i*16-11),T(i*16-10),T(i*16-9),T(i*16-8)];[T(i*16-7),T(i*16-6),T(i*16-5),T(i*16-4)];[T(i*16-3),T(i*16-2),T(i*16-1),T(i*16-0)]];
  i_index(i) = i;
end


% ----------------------------------------------------------------------------------------------------------------------
% Now estimate calibration parameters using LSQNONLIN.
% ----------------------------------------------------------------------------------------------------------------------

opt_options = optimset('lsqnonlin') ; % Use a least-square non-linear optimisation
opt_options.LargeScale = 'on';        % Set this as a Large scale problem
opt_options.Display = 'iter';         % Display results after each iteration
opt_options.MaxFunEvals = 100000;
opt_options.MaxIter = 100000;

start_params = [initialGuess(1), initialGuess(2), initialGuess(3), initialGuess(4), initialGuess(5), initialGuess(6), initialGuess(7), initialGuess(8), initialGuess(9), initialGuess(10)];
H = ones(size(start_params));

% Now run optimisation algorithm
[final_params, sumsqs, residuals, exitflag, output, lambda, J] = lsqnonlin(@CompCalResidual,H.*start_params,[],[],opt_options, tMr_matrices, pin_positions);

disp('Final residual (mm):');
%%disp(rms(residuals));
disp('Final calibration (image to tracker) matrix:');
rMi = Comp_RigidBody_Matrix(final_params);
disp(rMi);
disp('Final image scaling parameter (mm/pixel):');
disp(final_params(10));
    
% Find outliers
S = diag([final_params(10) final_params(10) 1 1]);
pts_r = [];
for i = 1:size(pin_positions,1)
    pts_r = [pts_r tMr_matrices{i,1}*rMi*S*pin_positions{i,1}];
end
  
D = pts_r - repmat(median(pts_r,2),1,size(pts_r,2));
D = magc(D(1:3,:));
i_outliers = i_index(find(D>3))

plot3d(pts_r,1,'.');

       
 


   
   
   
