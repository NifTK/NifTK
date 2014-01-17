function [i_outliers] = niftkUltrasoundPinCalibrationOutliers(final_params, rMi, tMr_matrices, pin_positions, i_index)

i_outliers = [];
S = diag([final_params(10) final_params(10) 1 1]);
pts_r = [];
for i = 1:size(pin_positions,1)
    pts_r = [pts_r tMr_matrices{i,1}*rMi*S*pin_positions{i,1}];
end
  
D = pts_r - repmat(median(pts_r,2),1,size(pts_r,2));
D = magc(D(1:3,:));

i_outliers = i_index(find(D>3));
plot3d(pts_r,1,'.');
       
 


   
   
   
