#!/bin/bash

set -x

current_dir=`dirname $0`

output_format=$1
fixed_image=$2
fixed_image_mask=$3
dilation=$4
symmetric=$5
dof=$6
scaling_using_skull=$7
similarity=$8
ajc=$9
region=${10}
interpolation=${11}
skip_registration=${12}
starting_arg=13

if [ "${MY_TEMP}" == "" ]
then
  temp_dir_prefix=~/temp/
else
  temp_dir_prefix=${MY_TEMP}
fi   

tmp_dir=`mktemp -d -q ~/temp/__affine_groupwise_reg.XXXXXX`
function cleanup
{
  echo "Cleaning up..."
  rm -rf  ${tmp_dir}
}
trap "cleanup" EXIT SIGINT SIGTERM SIGKILL 


fixed_image_mask_img=${tmp_dir}/`basename ${fixed_image_mask}.img`
makemask ${fixed_image} ${fixed_image_mask} ${fixed_image_mask_img}

moving_image_and_dof=""
moving_image_mask_and_dof=""
dilation_flag=""
symmetric_flag="-sym_midway"
number_of_images=1
ajc_flag=""
symmetric_ajc_flag="0"

if [ ${dilation} \> 0 ] 
then 
  dilation_flag="-d ${dilation}"
fi 

if [ "${symmetric}" == "no" ]
then
  symmetric_flag=""
else
  symmetric_flag=-"${symmetric}"
fi 

if [ "${ajc}" == "yes" ]
then 
  ajc_flag="-ajc"
  symmetric_ajc_flag="1"
fi 


for (( arg=${starting_arg}; arg<=$#; arg+=2 ))
do
  (( i=arg-starting_arg ))
  output=`printf ${output_format} ${i}`
  (( img_arg=arg ))
  moving_image=${!img_arg}
  (( mask_arg=arg+1 ))
  moving_image_mask=${!mask_arg}
  
  if [ "${region}" == "yes" ]
  then
    moving_image_mask_img=${tmp_dir}/`basename ${moving_image_mask}.img`
    makemask ${moving_image} ${moving_image_mask} ${moving_image_mask_img}
  fi
  
  if [ "${skip_registration}" != "yes" ]
  then 
  
  if [ "${region}" == "yes" ] 
  then 
    # Affine registration to align all the images to a randomly chosen fixed image. 
    niftkAffine \
      -ti ${fixed_image} -si ${moving_image} \
      -tm ${fixed_image_mask_img} -sm ${moving_image_mask_img} \
      -ot ${output}_affine_init.dof \
      -ri 2 -fi 3 -s 4 -tr 2 -o 7 -ln 4 -rmin 0.1 -rmax 2 -stl 0 -spl 2 ${symmetric_flag} ${dilation_flag} -wsim 2 -pptol 0.001
      #-ri 2 -fi 3 -s ${similarity} -tr 2 -o 6 -ln 3 -rmin 1 -stl 0 -spl 0 ${symmetric_flag}
      
    if [ "${scaling_using_skull}" == "yes" ] 
    then 
      fixed_image_brain_and_skull=${tmp_dir}/fixed_image_brain_and_skull.img
      fixed_image_brain=${tmp_dir}/fixed_image_brain.img
      fixed_image_skull=${tmp_dir}/fixed_image_skull.img
      fixed_image_skull_region=${tmp_dir}/fixed_image_skull
      niftkDilate -i ${fixed_image_mask_img} -o ${fixed_image_brain_and_skull} -it 24 -b 0 -d 255
      niftkDilate -i ${fixed_image_mask_img} -o ${fixed_image_brain} -it 2 -b 0 -d 255
      niftkSubtract -i ${fixed_image_brain_and_skull} -j ${fixed_image_brain} -o ${fixed_image_skull}
      makeroi -img ${fixed_image_skull} -out ${fixed_image_skull_region} -alt 1
      makemask ${fixed_image} ${fixed_image_skull_region} ${fixed_image_skull} -k -bpp 16
      
      moving_image_brain_and_skull=${tmp_dir}/moving_image_brain_and_skull.img
      moving_image_brain=${tmp_dir}/moving_image_brain.img
      moving_image_skull=${tmp_dir}/moving_image_skull.img
      moving_image_skull_region=${tmp_dir}/moving_image_skull
      niftkDilate -i ${moving_image_mask_img} -o ${moving_image_brain_and_skull} -it 24 -b 0 -d 255
      niftkDilate -i ${moving_image_mask_img} -o ${moving_image_brain} -it 2 -b 0 -d 255
      niftkSubtract -i ${moving_image_brain_and_skull} -j ${moving_image_brain} -o ${moving_image_skull}
      makeroi -img ${moving_image_skull} -out ${moving_image_skull_region} -alt 1
      makemask ${moving_image} ${moving_image_skull_region} ${moving_image_skull} -k -bpp 16
      
      niftkAffine \
        -ti ${fixed_image_skull} -si ${moving_image_skull} \
        -tm ${fixed_image_skull} \
        -sm ${moving_image_skull} \
        -ot ${output}_affine_first.dof \
        -it ${output}_affine_init.dof \
        -ri 2 -fi 3 -s ${similarity} -tr ${dof} -o 6 -ln 1 -rmin 0.01 ${symmetric_flag} -wsim 2 -pptol 0.001
    else
      niftkAffine \
        -ti ${fixed_image} -si ${moving_image} \
        -tm ${fixed_image_mask_img} \
        -sm ${moving_image_mask_img} \
        -ot ${output}_affine_first.dof \
        -it ${output}_affine_init.dof \
        -ri 2 -fi 3 -s ${similarity} -tr ${dof} -o 7 -ln 1 -rmin 0.001 -rmax 1 ${symmetric_flag} ${dilation_flag} -wsim 2 -pptol 0.0001
        #-ri 2 -fi 3 -s ${similarity} -tr ${dof} -o 6 -ln 1 -rmin 0.01 ${symmetric_flag} ${dilation_flag}
    fi 
    
  else # if [ "${region}" == "no" ] 
  
      niftkAffine \
        -ti ${fixed_image} -si ${moving_image} \
        -ot ${output}_affine_first.dof \
        -ri 2 -fi 3 -s ${similarity} -tr ${dof} -o 6 -ln 3 -rmax 1 -rmin 0.01 ${symmetric_flag} -wsim 2 -pptol 0.001
        
  fi 
    
     
  if [ ${dof} == 2 ] 
  then
     mv ${output}_affine_first.dof ${output}_affine_second.dof
  else
    niftkAffine \
      -ti ${fixed_image} -si ${moving_image} \
      -tm ${fixed_image_mask_img} \
      -sm ${moving_image_mask_img} \
      -ot ${output}_affine_second.dof \
      -it ${output}_affine_first.dof \
      -ri 2 -fi 3 -s ${similarity} -tr 2 -o 7 -ln 1 -rmin 0.001 -rmax 0.5 ${symmetric_flag} -d 2 -wsim 2 -pptol 0.0001
      #-ri 2 -fi 3 -s ${similarity} -tr 2 -o 6 -ln 2 -rmax 0.5 -rmin 0.01 ${symmetric_flag} -d 2
  fi 
  
  fi # endif [ "${skip_registration}" != "yes" ]
     
  (( number_of_images++ ))
  moving_image_and_dof=${moving_image_and_dof}" ${moving_image} ${output}_affine_second.dof"
  moving_image_mask_and_dof=${moving_image_mask_and_dof}" ${moving_image_mask_img} ${output}_affine_second.dof"
  
done

if [ "${symmetric}" == "no" ] 
then 

  for (( arg=${starting_arg}; arg<=$#; arg+=2 ))
  do
    (( i=arg-starting_arg ))
    output=`printf ${output_format} ${i}`
    (( img_arg=arg ))
    moving_image=${!img_arg}
    (( mask_arg=arg+1 ))
    moving_image_mask=${!mask_arg}
    
    moving_image_mask_img=${tmp_dir}/`basename ${moving_image_mask}.img`
    
    resliced_image=${output}-transformed.hdr
    mask_img=${output}-mask-transformed.img 
    output_mask=${output}-transformed-mask 
    
    niftkTransformation -ti ${fixed_image} -o ${resliced_image} -si ${moving_image} -j ${interpolation} -g ${output}_affine_second.dof ${ajc_flag} 
    niftkAbsImageFilter -i ${resliced_image} -o ${resliced_image}
    
    niftkTransformation -ti ${fixed_image} -o ${mask_img} -si ${moving_image_mask_img} -j 2 -g ${output}_affine_second.dof  ${ajc_flag}
    makeroi -img ${mask_img} -out ${output_mask} -alt 128
  done

else

  if [ "${symmetric}" == "sym" ] 
  then
  
    # Transform them to the affine "mid-point". 
    itkComputeInitialAffineAtlas \
      ${output_format}-average-affine.hdr \
      ${fixed_image} \
      4 0 ${symmetric_ajc_flag} \
      ${moving_image_and_dof}
    
    average_image=`printf ${output_format}-average-affine.hdr 999`
    niftkAbsImageFilter -i ${average_image} -o ${average_image}
      
    if [ "${region}" == "yes" ]     
    then 
      # Transform the mask to the affine "mid-point". 
      itkComputeInitialAffineAtlas \
        ${output_format}-mask.hdr \
        ${fixed_image_mask_img} \
        2 0 ${symmetric_ajc_flag} \
        ${moving_image_mask_and_dof}
        
      for (( i=0; i<${number_of_images}; i++ ))
      do
        resliced_image=`printf ${output_format}-average-affine.hdr ${i}`
        mask_img=`printf ${output_format}-mask.img ${i}`
        output_mask=`printf ${output_format}-average-affine-mask ${i}`
        
        niftkAbsImageFilter -i ${resliced_image} -o ${resliced_image}
        makeroi -img ${mask_img} -out ${output_mask} -alt 128
      done
    fi 
    
  else
    
    for (( arg=${starting_arg}; arg<=$#; arg+=2 ))
    do
      (( i=arg-starting_arg ))
      output=`printf ${output_format} ${i}`
      (( img_arg=arg ))
      moving_image=${!img_arg}
      (( mask_arg=arg+1 ))
      moving_image_mask=${!mask_arg}
      
      moving_image_mask_img=${tmp_dir}/`basename ${moving_image_mask}.img`
    
      resliced_image=${output}-0-transformed.hdr
      mask_img=${output}-0-mask-transformed.img 
      output_mask=${output}-0-transformed-mask 
      niftkTransformation -ti ${fixed_image} -o ${resliced_image} -j ${interpolation} -g ${output}_affine_second.dof ${ajc_flag} -sym_midway ${fixed_image} ${moving_image} -invertAffine
      niftkAbsImageFilter -i ${resliced_image} -o ${resliced_image}
      niftkTransformation -ti ${fixed_image_mask_img} -o ${mask_img} -j 2 -g ${output}_affine_second.dof  ${ajc_flag} -sym_midway ${fixed_image} ${moving_image} -invertAffine
      makeroi -img ${mask_img} -out ${output_mask} -alt 128
      
      resliced_image=${output}-1-transformed.hdr
      mask_img=${output}-1-mask-transformed.img 
      output_mask=${output}-1-transformed-mask 
      niftkTransformation -ti ${moving_image} -o ${resliced_image} -j ${interpolation} -g ${output}_affine_second.dof ${ajc_flag} -sym_midway ${fixed_image} ${moving_image}
      niftkAbsImageFilter -i ${resliced_image} -o ${resliced_image}
      niftkTransformation -ti ${moving_image_mask_img} -o ${mask_img} -j 2 -g ${output}_affine_second.dof  ${ajc_flag} -sym_midway ${fixed_image} ${moving_image}
      makeroi -img ${mask_img} -out ${output_mask} -alt 128
      
    done
  
  fi 
  
fi   


rm -rf ${tmp_dir}






