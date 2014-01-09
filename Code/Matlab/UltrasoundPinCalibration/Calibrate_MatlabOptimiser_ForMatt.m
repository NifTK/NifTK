% Calibrate_MatlabOptimiser.m
% Dean Barratt, 25/2/2008
% Computational Imaging Science Group, Imaging Sciences, King's College, London
% 
% 
% Compute the calibration matrix for the 3DUS system from data collected with a pin phantom.
% The head of the pin forms an invariant point, and is imaged using ultrasound. Two data files
% are required to run this program: one containing a set of position vectors for each scan of the
% pin. The second contains the corresponding transformation matrices which specify the position and
% orientation of a sensor fixed to the ultrasound probe measured using a tracking device. Therefore,
% each pin scan generates one pin position (relative to the image co-ordinate system, I) and one
% transformation matrix (tMr - see below)
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
% the Levenberg-Marquardt algorithm. 9 parameters are estimated: the x,y, and z components of the translation vector
% of vMt, and 3 rotation angles and the x,y, and z components of the translation of rMi.
% In this implementation The Jacobian is computed analytically (see Comp_Jacobian.m).

%%%%%%%%%%%%%%% Load tMr matrices from a file %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%file_selection = {'*.track','Tracking Files (*.track)'};
%[Filename, Pathname] = uigetfile(file_selection,'Select .track file');   
%if Filename == 0  % Return if Cancel is pressed
%   return;
%end

path_track = 'matrices.track';
file_id = fopen(path_track, 'r');
matrix_US2REF = {};

if (file_id ~= -1)  % If file exists ...
   j = 1;
   image_path = '-'
   while ~isempty(image_path)
        
        % Read image path, ignoring spaces
        line = textscan(file_id, '%s',1,'delimiter','\n','whitespace','');
        image_path = cell2mat(line{1});
        
        if ~isempty(image_path)
                % Find the last '\' character in path and extract image filename
                slash_places = strfind(image_path,'\');
                if isempty(slash_places)
                    image_name = image_path;
                else
                    image_name = image_path(slash_places(end)+1:end);
                end
               % disp(image_name);
              
                % Read the number of IREDs visible on reference object
                line = textscan(file_id, '%*s %*c %d %*s %*s',1);
                n_IREDS_REF= line{1};
                
                % Read transformation for reference object (REF -> OPTOTRAK)
                [matrix_REF, nread] = fscanf(file_id, '%f',[4,4]);         
                matrix_REF = matrix_REF';
                
                % Read the number of IREDs visible on tracked object
                line = textscan(file_id, '%*s %*c %d %*s %*s',1);
                n_IREDS_TRACKED = line{1};
                
                % Read transformation for tracked object (TRACKED -> OPTOTRAK)
                [matrix_TRACKED, nread] = fscanf(file_id, '%f',[4,4]);         
                matrix_TRACKED = matrix_TRACKED';
 
                
                % Calculate TRACKED -> REFERENCE transformation
                matrix_US2REF{j,1} = image_name;
                matrix_US2REF{j,2} = inv(matrix_REF)*matrix_TRACKED;   
               
                % Set flag to signify the quality of tracking data
                if ((n_IREDS_REF > 3) & (n_IREDS_TRACKED >= 10))
                    matrix_US2REF{j,3} = true;
                else
                    matrix_US2REF{j,3} = false;
                end
                
                % Read blank line
                textscan(file_id, '\n');
                
                j = j+1;     
        end 
      
   end
   fclose(file_id);
   fprintf('\n %d tracking matrices read in', j-1);

else
    errordlg('Unable to open selected track file!');
    return;
end

N_track = j-1;

%%%%%%%%%%% Load pinhead positions from a file  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%[Filename, Pathname] = uigetfile('*.mat','Open pin Matlab file containing pin co-ordinates');  % Select filename
%if Filename == 0  % Return if Cancel is pressed
%   return;
%end
% Load pin co-ordinate data in 'control_points' and 'image_file_list'
load('control_points.mat');

% ----------------------------------------------------------------------------------------------------------------------

% Calibration of L7-4 probe for Steve Thompson, 2008
%T_offset = [ -205.5 -65.5 ];    % Pixel Offset for top left hand corner of us image
%US_image_scaling_init = 40/402; % mm/pixel

% Calibration of L15-12 probe for Tim Carter, 2008
%T_offset = [-252.5 -65.5 ];  % Pixel Offset for top left hand corner of us image
T_offset = [0 0 ];  % Pixel Offset for top left hand corner of us image
%US_image_scaling_init = 0.081522; % = 30/(434 mm/pixel
%US_image_scaling_init = 0.0825; % = 30/(434 mm/pixel
%US_image_scaling_init = 0.1666666;
US_image_scaling_init = -0.2073;

% ----------------------------------------------------------------------------------------------------------------------

N_pincoords = size(control_points,1)
if (N_pincoords ~= N_track)
   errordlg('Mismatch between number of tracking matrices and US images! - check data files');
   return;
else
    N = N_track;
end
i_outliers = [];
tMr_matrices = {}; pin_positions = {}; i_index = []; j = 1;
for i = setdiff(1:N, i_outliers);  
    if (matrix_US2REF{i,3} & ~isempty(control_points{i}) )
        tMr_matrices{j,1} = matrix_US2REF{i,2};
        pin_positions{j,1} = [(T_offset + control_points{i}(:,1)') 0 1]'; % NB: Pixel co-ords!
        i_index(j) = i;
        j = j+1;
    end
end


% ----------------------------------------------------------------------------------------------------------------------

% Now estimate calibration parameters using LSQNONLIN

opt_options = optimset('lsqnonlin') ; % Use a least-square non-linear optimisation
opt_options.LargeScale = 'on';       % Set this as a Large scale problem
opt_options.Display = 'iter';         % Display results after each iteration
opt_options.MaxFunEvals = 100000;
opt_options.MaxIter = 100000;
  
%start_params = [0 0   0   0   0   0  150     0     -70 ]; %zeros(1,9); % Initial paramater estimate: 
start_params = [200 -10   0   0   0   20  545 237 -1820 US_image_scaling_init ]; %zeros(1,9); % Initial paramater estimate: 
                  %  [rMi_x, rMi_y, rMi_z, rMi_alpha, rMi_beta, rMi_gamma, vMt_x, vMt_y, vMt_z, s]

    H = ones(size(start_params));

    % Now run optimisation algorithm
    [final_params, sumsqs, residuals,exitflag,output,lambda,J] = lsqnonlin(@CompCalResidual,H.*start_params,[],[],...
     opt_options, tMr_matrices, pin_positions);    % Invoke optimizer

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

       
 


   
   
   
