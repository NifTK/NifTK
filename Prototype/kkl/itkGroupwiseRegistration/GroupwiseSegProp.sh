#!/bin/bash

if [ -z ${MIDAS_BIN} ]
then
  MIDAS_BIN=/var/drc/software/64bit/midas/bin
fi 

current_dir=`dirname $0`
  exe_path=${current_dir}/../../../../NifTK-build/bin/


atlas=$1
fixed_image_midas_mask=$2
resampling=$3
fixed_image=$4
fixed_image_affine_name=$5
fixed_image_affine_dof=$6
fixed_image_nreg_dof=$7

abs_atlas=${atlas%.*}-abs.img
fixed_image_affine_mask=${fixed_image_midas_mask%.*}_atlas_affine_0.img
fixed_image_affine_mask_format=${fixed_image_midas_mask%.*}_atlas_affine_%i.img
fixed_image_mask=${fixed_image_midas_mask}_fixed_mask.img
fixed_image_nreg_mask=${fixed_image_midas_mask%.*}_atlas_nreg.img
fixed_image_nreg_midas_mask=${fixed_image_midas_mask}_atlas_nreg
fixed_image_nreg_temp_mask=${fixed_image_midas_mask}_atlas_nreg_temp.img
individual_mask_format=${fixed_image%.*}-final_mask_%i.img


if [ ! -f ${atlas} ]
then 
  echo "${atlas} not found!"
  exit
fi
if [ ! -f ${fixed_image_midas_mask} ]
then 
  echo "${fixed_image_midas_mask} not found!"
  exit
fi

is_resample=`echo " ${resampling} > 0"|bc`
if [ ${is_resample} != 0 ]
then
  resample_flag="-resample ${resampling}"
fi  


shift 3
number_of_image=0
while [ $# -gt 0 ]
do
  individual_original_image_name=$1
  individual_affine_image_name=$2
  individual_affine_dof_name=$3
  individual_nreg_dof_name=$4
  
  individual_original_image[$number_of_image]=${individual_original_image_name}
  individual_mask[$number_of_image]=`printf ${individual_mask_format} ${number_of_image}`
  individual_original_image_final_mask[$number_of_image]=${individual_original_image_name%.*}_mask_final.img
  individual_midas_mask[$number_of_image]=${individual_original_image_name%.*}_mask
  individual_nreg_mask[$number_of_image]=${individual_original_image_name%.*}_nreg_mask.img
  individual_affine_dof[$number_of_image]=${individual_affine_dof_name}
  individual_nreg_dof[$number_of_image]=${individual_nreg_dof_name}
  individual_nreg_invert_dof[$number_of_image]=${individual_nreg_dof_name%.*}_invert.dof
  individual_nreg_accurate_invert_dof[$number_of_image]=${individual_nreg_dof_name%.*}_accurate_invert.dof
  individual_affine_image[$number_of_image]=${individual_affine_image_name}
  individual_nreg_image[$number_of_image]=${individual_original_image_name%.*}_accurate_nreg.img
  if [ ! -f ${individual_original_image[$number_of_image]} ]
  then 
    echo "${individual_original_image[$number_of_image]} not found!"
    exit
  fi
  if [ ! -f ${individual_affine_dof[$number_of_image]} ] && [ ${individual_affine_dof[$number_of_image]} != "dummy.dof" ]
  then 
    echo "${individual_affine_dof[$number_of_image]} not found!"
    exit
  fi
  if [ ! -f ${individual_nreg_dof[$number_of_image]} ]
  then 
    echo "${individual_nreg_dof[$number_of_image]} not found!"
    exit
  fi
  if [ ! -f ${individual_affine_image[$number_of_image]} ]
  then
    echo "${individual_affine_image[$number_of_image]} not found"
  fi 
  
  (( number_of_image++ ))
  shift 4
done


#
# Propagate from manual segmentation to the average atlas. 
#
# 1. Mask from Midas ROI. 
echo "Masking original mask..."
${MIDAS_BIN}/makemask ${fixed_image} ${fixed_image_midas_mask} ${fixed_image_mask}

# 2. Transform mask affinely.
echo "Transform the mask affinely..."
image_and_affine_dof=""
for (( i=1; i<number_of_image; i++ ))
do 
  image_and_affine_dof="${image_and_affine_dof} ${fixed_image_mask} ${individual_affine_dof[$i]}"
done
${exe_path}itkComputeInitialAffineAtlas ${fixed_image_affine_mask_format} \
                                              ${fixed_image_mask} \
                                              2 0 \
                                              ${image_and_affine_dof}
${exe_path}niftkThreshold -i ${fixed_image_affine_mask} -o ${fixed_image_affine_mask} -u 5000 -l 128 -in 256 -out 0 

# 3. Transform mask nonrigidly.      
echo "Transforming the mask to the average atlas..."
${exe_path}niftkTransformation -ti ${atlas} -si ${fixed_image_affine_mask} \
                                -o ${fixed_image_nreg_mask} \
                                -df ${fixed_image_nreg_dof} \
                                -j 2 
${exe_path}niftkThreshold -i ${fixed_image_nreg_mask} -o ${fixed_image_nreg_mask} -u 5000 -l 128 -in 256 -out 0 
#                                -g ${fixed_image_affine_dof} \
                                
# 4. Erode and conditional dilate. 
echo "Doing morphology..."
${exe_path}niftkAbsImageFilter -i ${atlas} -o ${abs_atlas}
${MIDAS_BIN}/makeroi -img ${fixed_image_nreg_mask} -out ${fixed_image_nreg_midas_mask} -alt 128
${MIDAS_BIN}/makemask ${abs_atlas} ${fixed_image_nreg_midas_mask} ${fixed_image_nreg_temp_mask} -e 1
${MIDAS_BIN}/makeroi -img ${fixed_image_nreg_temp_mask} -out ${fixed_image_nreg_midas_mask} -alt 128 
${MIDAS_BIN}/makemask ${abs_atlas} ${fixed_image_nreg_midas_mask} ${fixed_image_nreg_temp_mask} -cd 1 60 160
${MIDAS_BIN}/makeroi -img ${fixed_image_nreg_temp_mask} -out ${fixed_image_nreg_midas_mask} -alt 128 

${remove} -f ${abs_atlas} ${abs_atlas%.img}.hdr

#
# Propagate back to the original images. 
# 
# 1. Invert the nonrigid transform and transform the mask. 
for (( i=0; i<number_of_image; i++ ))
do 
  if [ ! -f ${individual_nreg_invert_dof[$i]} ]
  then
    echo "Inverting transform ${individual_nreg_dof[$i]} to ${individual_nreg_invert_dof[$i]}..."
    ${exe_path}itkInvertTransformation ${individual_nreg_dof[$i]} ${individual_nreg_invert_dof[$i]}
  fi
  
  echo "Transform mask back nonrigidly with rough estimation..."
  ${exe_path}niftkTransformation -ti ${individual_original_image[$i]} -si ${fixed_image_nreg_temp_mask} \
                                  -o ${individual_nreg_mask[$i]} \
                                  -df ${individual_nreg_invert_dof[$i]} \
                                  -j 2 
                                  
  ${exe_path}niftkFluid \
      -ti ${individual_affine_image[$i]} -si ${atlas} \
      -tm ${individual_nreg_mask[$i]} \
      -oi ${individual_nreg_image[$i]} \
      -to ${individual_nreg_accurate_invert_dof[$i]} \
      -it ${individual_nreg_invert_dof[$i]} \
      -d 4 \
      -ln 1 -fi 4 -ri 2 -is 0.7 -cs 1 -md 0.05 -mc 1.0e-9 -ls 1.0 -rs 1.0 -force ssdn -sim 4 ${resample_flag} \
      -hfl -1e6 -hfu 1e6 -hml -1e6 -hmu 1e6
      
  echo "Transform mask back nonrigidly..."
  ${exe_path}niftkTransformation -ti ${individual_original_image[$i]} -si ${fixed_image_nreg_temp_mask} \
                                  -o ${individual_nreg_mask[$i]} \
                                  -df ${individual_nreg_accurate_invert_dof[$i]} \
                                  -j 2 
                                  
  ${exe_path}niftkThreshold -i ${individual_nreg_mask[$i]} -o ${individual_nreg_mask[$i]} -u 5000 -l 128 -in 256 -out 0 
  
  echo "Release disk space by removing the dofs"
  ${remove} -f ${individual_nreg_invert_dof[$i]} ${individual_nreg_accurate_invert_dof[$i]}
  
done                                  

${remove} -f ${fixed_image_nreg_temp_mask} ${fixed_image_nreg_temp_mask%.img}.hdr
  
# 2. Transform it to the original image space. 
image_and_affine_dof=""
for (( i=1; i<number_of_image; i++ ))
do 
  image_and_affine_dof="${image_and_affine_dof} ${individual_nreg_mask[$i]} ${individual_affine_dof[$i]}"
done
echo "Transform mask back affinely..."
${exe_path}itkComputeInitialAffineAtlas ${individual_mask_format} \
                                         ${individual_nreg_mask[0]} \
                                         2 1 \
                                         ${image_and_affine_dof}
  
# 3. Erode and conditional dilate. 
echo "Doing morphology..."
for (( i=0; i<number_of_image; i++ ))
do 
  ${MIDAS_BIN}/makeroi -img ${individual_mask[$i]} -out ${individual_midas_mask[$i]} -alt 128
  ${MIDAS_BIN}/makemask ${individual_original_image[$i]} ${individual_midas_mask[$i]} ${individual_mask[$i]} -e 1
  ${MIDAS_BIN}/makeroi -img ${individual_mask[$i]} -out ${individual_midas_mask[$i]} -alt 128 
  ${MIDAS_BIN}/makemask ${individual_original_image[$i]} ${individual_midas_mask[$i]} ${individual_original_image_final_mask[$i]} -cd 2 60 160
  ${MIDAS_BIN}/makeroi -img ${individual_original_image_final_mask[$i]} -out ${individual_midas_mask[$i]} -alt 128 
done

for (( i=0; i<number_of_image; i++ ))
do 
  # remove ${fixed_image%.*}-final_mask_%i.*
  ${remove} -f ${individual_mask[$i]} ${individual_mask[$i]%.img}.hdr
  # remove ${individual_original_image_name%.*}_mask_final.*
  ${remove} -f ${individual_original_image_final_mask[$i]} ${individual_original_image_final_mask[$i]%.img}.hdr
  # remove ${fixed_image_midas_mask%.*}_atlas_affine_${i}.*
  ${remove} -f ${fixed_image_midas_mask%.*}_atlas_affine_${i}.img ${fixed_image_midas_mask%.*}_atlas_affine_${i}.hdr
  # remove ${individual_original_image_name%.*}_nreg_mask.* 
  ${remove} -f ${individual_nreg_mask[$i]} ${individual_nreg_mask[$i]%.img}.hdr
  # remove ${individual_original_image_name%.*}_accurate_nreg.* 
  ${remove} -f ${individual_nreg_image[$i]} ${individual_nreg_image[$i]%.img}.hdr
done

  
echo "Done. Now go and check your results... NOW!"  
  
  
  
  













