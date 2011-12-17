

atlas_image=$1 
atlas_image_midas_mask=$2
atlas_image_dilated_mask=${atlas_image%.*}_dilated_mask.img
atlas_image_mask=${atlas_image%.*}_mask.img
atlas_image_resampled=${atlas_image%.*}_resampled.img  
atlas_image_mask_resampled=${atlas_image%.*}_mask_resampled.img

target_image=$3 
target_image_affine_dof=${target_image%.*}_affine.dof
target_image_affine_image=${target_image%.*}_affine.img
target_image_affine_mask=${target_image%.*}_affine_mask.img
target_image_affine_midas_mask=${target_image%.*}_affine_mask
target_image_affine_dilated_mask=${target_image%.*}_affine_dilated_mask.img
target_image_resampled=${target_image%.*}_resampled.img
target_image_affine_dilated_mask_resampled=${target_image%.*}_affine_dilated_mask_resampled.img
target_image_nreg_dof=${target_image%.*}_nreg.dof
target_image_nreg_image=${target_image%.*}_nreg.img
target_image_nreg_mask=${target_image%.*}_nreg_mask.img
target_image_nreg_midas_mask=${target_image%.*}_nreg_mask
target_image_nreg_dilated=${target_image%.*}_nreg_dilated.img
target_image_kmeans=${target_image%.*}_nreg_kmeans.img
target_image_threshoold_gm_wm_mask=${target_image%.*}_threshold_gm_wm_mask.img
target_image_threshoold_gm_wm_midas_mask=${target_image%.*}_threshold_gm_wm_mask
target_image_high_intensity=${target_image%.*}_high_intensity.img
target_image_threshold_wm_mask=${target_image%.*}_threshold_wm_mask.img
target_image_threshold_wm_midas_mask=${target_image%.*}_threshold_wm_mask
target_image_threshold_wm_eroded_mask=${target_image%.*}_threshold_wm_eroded_mask.img
target_image_threshold_wm_connected_mask=${target_image%.*}_threshold_wm_connected_mask.img
target_image_distance_map=${target_image%.*}_distance_map.img
target_image_watershed=${target_image%.*}_watershed.img
target_image_distance_map_threshold=${target_image%.*}_distance_map_threshold.img
target_image_distance_map_connected=${target_image%.*}_distance_map_connected.img
target_image_distance_map_connected_threshold=${target_image%.*}_distance_map_connected_threshold.img
target_image_distance_map_recovered=${target_image%.*}_distance_map_recovered.img
target_image_distance_map_recovered_threshold=${target_image%.*}_distance_map_recovered_threshold.img
target_image_distance_map_recovered_threshold_midas_mask=${target_image%.*}_distance_map_recovered_threshold
target_image_accurate_kmeans=${target_image%.*}_accurate_kmeans.img
target_image_distance_map_recovered_threshold_dilated=${target_image%.*}_distance_map_recovered_threshold_dilated.img
target_image_distance_map_recovered_threshold_dilated_midas_mask=${target_image%.*}_distance_map_recovered_threshold_dilated

resampling=0.9375


# 
# 1. Register affinely the atlas and the target image. 
#
echo makemask ${atlas_image} ${atlas_image_midas_mask} ${atlas_image_dilated_mask} -d 5
echo makemask ${atlas_image} ${atlas_image_midas_mask} ${atlas_image_mask}

echo /var/NOT_BACKUP/work/NifTK-build/bin/niftkAffine \
    -ti ${target_image} -si ${atlas_image} \
    -sm ${atlas_image_dilated_mask} \
    -ot ${target_image_affine_dof} \
    -oi ${target_image_affine_image} \
    -ri 1 -fi 3 -s 9 -tr 2 -o 6 -ln 3 -rmin 1 -sym
  
echo /var/NOT_BACKUP/work/NifTK-build/bin/niftkAffine \
    -ti ${target_image} -si ${atlas_image} \
    -sm ${atlas_image_dilated_mask} \
    -ot ${target_image_affine_dof} \
    -oi ${target_image_affine_image} \
    -it ${target_image_affine_dof} \
    -ri 1 -fi 3 -s 9 -tr 3 -o 5 -ln 3 -rmin 0.5 -sym
    
echo /var/NOT_BACKUP/work/NifTK-build/bin/niftkTransformation \
    -ti ${target_image} \
    -si ${atlas_image_mask} \
    -o ${target_image_affine_mask}  \
    -g ${target_image_affine_dof} -j 1
    
echo makeroi -img ${target_image_affine_mask} -out ${target_image_affine_midas_mask} -alt 127

echo /var/NOT_BACKUP/work/NifTK-build/bin/niftkTransformation \
    -ti ${target_image} \
    -si ${atlas_image_dilated_mask} \
    -o ${target_image_affine_dilated_mask}  \
    -g ${target_image_affine_dof} -j 1
    


#
# 2. Nonrigidly register the atlas and the target image. 
#
echo ~/work/NifTK-build/bin/itkResampleTest ${atlas_image} ${atlas_image_resampled} 3 ${resampling} ${resampling} ${resampling}
echo ~/work/NifTK-build/bin/itkResampleTest ${atlas_image_mask} ${atlas_image_mask_resampled} 1 ${resampling} ${resampling} ${resampling}
echo ~/work/NifTK-build/bin/itkResampleTest ${target_image} ${target_image_resampled} 3 ${resampling} ${resampling} ${resampling}
echo ~/work/NifTK-build/bin/itkResampleTest ${target_image_affine_dilated_mask} ${target_image_affine_dilated_mask_resampled} 1 ${resampling} ${resampling} ${resampling}

echo ${HOME}/work/NifTK-build/bin/niftkFluid \
   -ti ${target_image_resampled} -si ${atlas_image_resampled} \
   -tm ${target_image_affine_dilated_mask_resampled}  \
   -oi ${target_image_nreg_image} \
   -to ${target_image_nreg_dof} \
   -adofin ${target_image_affine_dof} \
   -ln 2 -fi 2 -ri 2 -is 0.5 -cs 1 -md 0.1 -ls 0.2 -rs 0.5 -force nmi -sim 9 -rescale 0 1000 -stl 0 -spl 0
   
#   -ln 3 -fi 2 -ri 2 -is 0.5 -cs 1 -md 0.01 -ls 0.2 -rs 0.5 -force ssd -sim 4 -rescale 0 1000 -stl 0 -spl 1

echo ${HOME}/work/NifTK-build/bin/niftkTransformation \
    -ti ${target_image_resampled} \
    -si ${atlas_image_mask_resampled} \
    -o ${target_image_nreg_mask}  \
    -g ${target_image_affine_dof} \
    -df ${target_image_nreg_dof} -j 1
    
#~/work/NifTK-build/bin/itkResampleTest ${target_image_nreg_mask} ${target_image_nreg_mask} 1 0.937500  0.937500  1.500000
echo makeroi -img ${target_image_nreg_mask} -out ${target_image_nreg_midas_mask} -alt 127
    
#
# Distance transform and morphology based brain segmentation. 
#
if [ 1 == 0 ]
then 

# Dilate it to make sure we cover the whole brain. 
makemask ${target_image} ${target_image_nreg_midas_mask} ${target_image_nreg_dilated} -d 4 -k -bpp 16
    
mean_intensity=`imginfo  ${target_image} -av -roi ${target_image_nreg_midas_mask}`
echo "mean_intensity="${mean_intensity}
wm_intensity=`echo "1.2*${mean_intensity}"|bc`
gm_intensity=`echo "0.7*${mean_intensity}"|bc`
csf_intensity=`echo "${mean_intensity}/2"|bc`

# k-means 
initial_means=`/var/NOT_BACKUP/work/NifTK-build/bin/itkKmeansClassifierTest ${target_image} ${target_image_nreg_dilated} ${target_image_kmeans} 1 3 ${csf_intensity} ${gm_intensity} ${wm_intensity}`
     
/var/NOT_BACKUP/work/NifTK-build/bin/niftkThreshold -i ${target_image_kmeans} -o ${target_image_threshoold_gm_wm_mask} -u 10 -l 2 -in 255 -out 0 
makeroi -img ${target_image_threshoold_gm_wm_mask} -out ${target_image_threshoold_gm_wm_midas_mask} -alt 127


echo "initial_means="${initial_means}
wm_mean=`echo ${initial_means} |awk '{printf $5}'`
wm_std=`echo ${initial_means} |awk '{printf $6}'`
wm_upper=`echo "${wm_mean}+3*${wm_std}"|bc`
echo "wm_upper="${wm_upper}
# Take out the really high intensity (fat) bits, mostly in the skull. 
/var/NOT_BACKUP/work/NifTK-build/bin/niftkThreshold -i ${target_image} -o ${target_image_high_intensity} -u 32767 -l ${wm_upper} -in 255 -out 0
/var/NOT_BACKUP/work/NifTK-build/bin/niftkSubtract -i ${target_image_threshoold_gm_wm_mask} -j ${target_image_high_intensity} -o ${target_image_threshoold_gm_wm_mask}
/var/NOT_BACKUP/work/NifTK-build/bin/niftkThreshold -i ${target_image_threshoold_gm_wm_mask} -o ${target_image_threshoold_gm_wm_mask} -u 32767 -l 127 -in 255 -out 0 
makeroi -img ${target_image_threshoold_gm_wm_mask} -out ${target_image_threshoold_gm_wm_midas_mask} -alt 127

# Get the WM mask. 
/var/NOT_BACKUP/work/NifTK-build/bin/niftkThreshold -i ${target_image_kmeans} -o ${target_image_threshold_wm_mask} -u 10 -l 3 -in 255 -out 0 
makeroi -img ${target_image_threshold_wm_mask} -out ${target_image_threshold_wm_midas_mask} -alt 127
makemask ${target_image_threshold_wm_mask} ${target_image_threshold_wm_midas_mask} ${target_image_threshold_wm_eroded_mask} -e 1 -bpp 16
/var/NOT_BACKUP/work/NifTK-build/bin/itkConnectedComponentTest \
    ${target_image_threshold_wm_eroded_mask} \
    ${target_image_threshold_wm_eroded_mask} \
    ${target_image_threshold_wm_connected_mask} 1 256 1 1
/var/NOT_BACKUP/work/NifTK-build/bin/niftkThreshold  -i ${target_image_threshold_wm_connected_mask} -o ${target_image_threshold_wm_connected_mask} -u 1 -l 1 -in 255 -out 0
    
# Distance transform of the GM/WM and CSF interface. 
# 1st transform to exclude the CSF and take out the brain and 2nd transform to recover CSF. 
/var/NOT_BACKUP/work/NifTK-build/bin/itkDistanceMapTest \
    ${target_image_threshoold_gm_wm_mask} \
    ${target_image_distance_map} \
    ${target_image_watershed} 0.01 0
/var/NOT_BACKUP/work/NifTK-build/bin/niftkThreshold \
    -i ${target_image_distance_map} \
    -o ${target_image_distance_map_threshold} \
    -u 32767 -l -1.874 -in 0 -out 255
/var/NOT_BACKUP/work/NifTK-build/bin/itkConnectedComponentTest \
    ${target_image_distance_map_threshold} \
    ${target_image_distance_map_threshold} \
    ${target_image_distance_map_connected} 1 256 1 1
/var/NOT_BACKUP/work/NifTK-build/bin/niftkThreshold  \
    -i ${target_image_distance_map_connected}  \
    -o ${target_image_distance_map_connected_threshold}  \
    -u 1 -l 1 -in 255 -out 0 
/var/NOT_BACKUP/work/NifTK-build/bin/itkDistanceMapTest \
    ${target_image_distance_map_connected_threshold} \
    ${target_image_distance_map_recovered} \
    ${target_image_watershed} 0.01 0
/var/NOT_BACKUP/work/NifTK-build/bin/niftkThreshold \
    -i ${target_image_distance_map_recovered} \
    -o ${target_image_distance_map_recovered_threshold} \
    -u 32767 -l 1.875 -in 0 -out 255
    
    
    
# Add WM mask.                                                          
/var/NOT_BACKUP/work/NifTK-build/bin/itkAddImageTest \
    ${target_image_distance_map_recovered_threshold} \
    ${target_image_threshold_wm_connected_mask} \
    ${target_image_distance_map_recovered_threshold}
makeroi -img ${target_image_distance_map_recovered_threshold} -out ${target_image_distance_map_recovered_threshold_midas_mask} -alt 127

fi 
    
# Get the CSF, GM and WM means for the conditional dilation.     
mean_intensity=`imginfo  ${target_image} -av -roi ${target_image_distance_map_recovered_threshold_midas_mask}`
wm_intensity=`echo "${mean_intensity}"|bc`
gm_intensity=`echo "4*${mean_intensity}/5"|bc`
echo ${gm_intensity},${wm_intensity}
means=`/var/NOT_BACKUP/work/NifTK-build/bin/itkKmeansClassifierTest ${target_image} ${target_image_distance_map_recovered_threshold} ${target_image_accurate_kmeans} 1 2 ${gm_intensity} ${wm_intensity}`
echo ${means}
gm_mean=`echo ${means} |awk '{printf $1}'`
gm_std=`echo ${means} |awk '{printf $2}'`
wm_mean=`echo ${means} |awk '{printf $3}'`
wm_std=`echo ${means} |awk '{printf $4}'`
lower=`echo "((${gm_mean}-4*${gm_std})*100)/${mean_intensity}"|bc`
upper=`echo "((${wm_mean}+4*${wm_std})*100)/${mean_intensity}"|bc`
echo ${lower}, ${upper}
harsh_lower=`echo "((${gm_mean}-2*${gm_std})*100)/${mean_intensity}"|bc`
harsh_upper=`echo "((${wm_mean}+2*${wm_std})*100)/${mean_intensity}"|bc`
echo ${harsh_lower}, ${harsh_upper}

# Conditional dilation. 
makemask ${target_image} ${target_image_distance_map_recovered_threshold_midas_mask} ${target_image_distance_map_recovered_threshold_dilated} -cd 1 ${harsh_lower} ${harsh_upper}    
makeroi -img ${target_image_distance_map_recovered_threshold_dilated} -out ${target_image_distance_map_recovered_threshold_dilated_midas_mask} -alt 127
reginfo ${target_image_distance_map_recovered_threshold_dilated_midas_mask} -v ${target_image_distance_map_recovered_threshold_dilated}

for ((i=0; i<4; i++)) 
do 
  makemask ${target_image} ${target_image_distance_map_recovered_threshold_dilated_midas_mask} ${target_image_distance_map_recovered_threshold_dilated} -cd 1 ${harsh_lower} ${harsh_upper}    
  makeroi -img ${target_image_distance_map_recovered_threshold_dilated} -out ${target_image_distance_map_recovered_threshold_dilated_midas_mask} -alt 127
  reginfo ${target_image_distance_map_recovered_threshold_dilated_midas_mask} -v ${target_image_distance_map_recovered_threshold_dilated}
done
    
if [ 1 == 0 ]
then 
sf
fi 











