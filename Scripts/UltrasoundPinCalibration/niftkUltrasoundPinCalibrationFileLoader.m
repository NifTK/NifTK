function [trackingMatrices, ultrasoundPoints, iIndex] = niftkUltrasoundPinCalibrationFileLoader()
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Usage:
%   [trackingMatrices, ultrasoundPoints, iIndex] = niftkUltrasoundPinCalibrationFileLoader()
%
% This functions will ask for:
%
%   matrixFile   : a single file containing N tracking matrices as 4x4 matrices, one after the other.
%   pointFile    : a single file containing N points, with x and y on the same line and seperate
%                  points on each line, and in same order as matrices.
%
% and loads matrices and points into cell arrays and returns them.
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

file_id = fopen(strcat(matrixPathname,'/',matrixFilename), 'r');
[T,countT] = fscanf(file_id, '%f');
fclose(file_id);
fprintf('\n %d matrices read in...\n', countT/16);

file_id = fopen(strcat(pointPathname,'/',pointFilename), 'r');
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
trackingMatrices = {};
ultrasoundPoints = {};
iIndex = [];

for i = 1:N
  ultrasoundPoints{i,1} = transpose([P(i*2-1), P(i*2), 0, 1]);
  trackingMatrices{i,1}  = [[T(i*16-15),T(i*16-14),T(i*16-13),T(i*16-12)];[T(i*16-11),T(i*16-10),T(i*16-9),T(i*16-8)];[T(i*16-7),T(i*16-6),T(i*16-5),T(i*16-4)];[T(i*16-3),T(i*16-2),T(i*16-1),T(i*16-0)]];
  iIndex(i) = i;
end
   
