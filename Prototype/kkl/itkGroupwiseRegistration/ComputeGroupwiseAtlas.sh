#!/bin/bash

current_dir=`dirname $0`

  exe_path=${current_dir}/../../../../NifTK-build/bin/

atlas_format=$1
output_format=$2
max_diff_file=$3
fixed_image=$4
fixed_image_mask=$5
resampling=$6

if [ ! -f ${fixed_image} ]
then 
  echo ${fixed_image}" not found!"
  exit
fi
if [ ! -f ${fixed_image_mask} ]
then 
  echo ${fixed_image_mask}" not found!"
  exit
fi

is_resample=`echo " ${resampling} > 0"|bc`
resample_flag=""
if [ ${is_resample} != 0 ]
then
  resample_flag="-resample ${resampling}"
fi  

shift 6
number_of_moving_image=0

stop=0
while [ $# -gt 0 ]
do
  moving_image[$number_of_moving_image]=$1
  initial_dof[$number_of_moving_image]=$2
  if [ ! -f ${moving_image[$number_of_moving_image]} ]
  then 
    echo ${moving_image[$number_of_moving_image]}" not found!"
    exit
  fi
  if [ ! -f ${initial_dof[$number_of_moving_image]} ]
  then 
    echo ${initial_dof[$number_of_moving_image]}" not found!"
    exit
  fi
  
  
  (( number_of_moving_image++ ))
  shift 2
done


stop=0
count=0
while [[ ${stop} != 1 &&  ${count} -lt 6 ]]
do
  dofs=""
  for (( i=0; i<${number_of_moving_image}; i++ ))
  do
    output=`printf ${output_format} ${i}`
    
    ${exe_path}niftkFluid \
      -ti ${fixed_image} -si ${moving_image[$i]} \
      -tm ${fixed_image_mask} \
      -oi ${output}.hdr \
      -to ${output}.dof \
      -it ${initial_dof[$i]} \
      -ln 1 -fi 4 -ri 2 -is 0.7 -cs 1 -md 0.05 -mc 1.0e-9 -ls 1.0 -rs 1.0 -force ssdn -sim 4 ${resample_flag} \
      -hfl -1e6 -hfu 1e6 -hml -1e6 -hmu 1e6
      
    dofs=${dofs}" ${output}.dof"
    
    initial_dof[$i]=${output}.dof
  done
  
  ${exe_path}itkComputeNewGroupwiseTransformations ${output_format}.dof ${max_diff_file} ${dofs}
  
  # Create the new atlas. 
  new_images=""
  for (( i=0; i<${number_of_moving_image}; i++ ))
  do
    output=`printf ${output_format} ${i}`
    ${exe_path}niftkTransformation -ti ${fixed_image} -si ${moving_image[$i]} -o ${output}-temp.hdr -df ${output}.dof -j 4
    new_images=${new_images}" "${output}-temp.hdr
  done
  atlas=`printf ${atlas_format} ${count}`.hdr
  image_diff_file=`printf ${atlas_format} ${count}`_diff.txt
  ${exe_path}itkSimpleAverage ${atlas} ${fixed_image_mask} ${new_images}
  
  ${exe_path}itkComputeImageDifference ${fixed_image} ${atlas} > ${image_diff_file}
  image_diff=`cat ${image_diff_file}`
  fixed_image=${atlas}
  
  # If, on average, we have less than about 0.1% difference between the two images we stop. 
  stop=`echo "${image_diff#-} < 80000" | bc`
  
  (( count++ ))
done

