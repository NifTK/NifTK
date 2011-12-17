#!/bin/bash 
  current_dir=`dirname $0`
  
  exe_path=${current_dir}/../../../../NifTK-build/bin/

  atlas_image=$1
  atlas_image_dilated_mask=$2
  use_manual_mask=$3
  each_line=$4
  input_dir=$5
  output_dir=$6
  
  #export remove="ls"
  export remove="rm"
  
#  echo ${each_line}
#  echo ${input_dir}
#  echo ${output_dir}
  
  number_of_fields=`echo "${each_line}"|awk -F, '{printf NF}'`
  number_of_images=`echo "${number_of_fields}/2" |bc`
  
  for (( i=0; i<${number_of_images}; i++ )) 
  do
    awk_index=$(( i*2+1 ))
    input_image[$i]=`echo "${each_line}"|awk -F, -v j=${awk_index} '{printf $j}'`
    awk_index=$(( i*2+2 ))
    input_region[$i]=`echo "${each_line}"|awk -F, -v j=${awk_index} '{printf $j}'`
  done
  for (( i=0; i<${number_of_images}; i++ )) 
  do
    # echo ${input_image[$i]}, ${input_region[$i]}
    local_input_image[$i]=${input_dir}/`basename ${input_image[$i]}`
    local_input_region[$i]=${input_dir}/`basename ${input_region[$i]}_mask.img`
  done

  midas_fixed_image_img_basename=`basename ${local_input_image[0]}`
  subject_id=${midas_fixed_image_img_basename:0:5}
  
  target_image_affine_dof=${subject_id}-mni-affine.dof
  target_image_affine_dilated_mask=${subject_id}-mni-affine-mask.img
  
  # 
  # 1. Register affinely the atlas and the target image. 
  #
  if [ ${use_manual_mask} = "1" ]
  then
  ${exe_path}niftkAffine \
    -ti ${local_input_image[0]} -si ${atlas_image} \
    -sm ${atlas_image_dilated_mask} \
    -ot ${target_image_affine_dof} \
    -ri 1 -fi 3 -s 9 -tr 2 -o 6 -ln 3 -rmin 1 -sym
  
  ${exe_path}niftkAffine \
    -ti ${target_image} -si ${atlas_image} \
    -sm ${atlas_image_dilated_mask} \
    -ot ${target_image_affine_dof} \
    -it ${target_image_affine_dof} \
    -ri 1 -fi 3 -s 9 -tr 3 -o 5 -ln 3 -rmin 0.5 -sym
    
  ${exe_path}niftkTransformation \
    -ti ${target_image} \
    -si ${atlas_image_dilated_mask} \
    -o ${target_image_affine_dilated_mask}  \
    -g ${target_image_affine_dof} -j 1
    
    local_input_region[0]=${target_image_affine_dilated_mask}
  fi 
  
  initial_affine_atlas_command_line="${local_input_image[0]} ${local_input_region[0]}"
  for (( i=1; i < number_of_images; i++ ))
  do 
    initial_affine_atlas_command_line="${initial_affine_atlas_command_line} ${local_input_image[$i]}"
  done
  
  initial_atlas_command_line=""
  for (( i=1; i < number_of_images; i++ ))
  do 
    initial_atlas_command_line="${initial_atlas_command_line} ${output_dir}/${subject_id}-${i}-average-affine.hdr"
  done
  groupwise_atlas_command_line=""
  for (( i=0; i < number_of_images; i++ ))
  do 
    groupwise_atlas_command_line="${groupwise_atlas_command_line} ${output_dir}/${subject_id}-${i}-average-affine.hdr ${output_dir}/${subject_id}-${i}-nreg-init.dof"
  done
  

  # Affine atlas. 
  ${current_dir}/ComputeInitialAffineAtlas.sh \
    ${output_dir}/${subject_id}-%i ${initial_affine_atlas_command_line}

  # Initial nonrigid atlas. 
  ${current_dir}/ComputeInitialAtlas.sh \
    ${output_dir}/${subject_id}-initial-atlas.hdr \
    ${output_dir}/${subject_id}-%i \
    ${output_dir}/${subject_id}-0-average-affine.hdr \
    ${output_dir}/${subject_id}-0-mask.hdr  \
    1 \
    ${initial_atlas_command_line}

  # Iteratively improve the atlas. 
  ${current_dir}/ComputeGroupwiseAtlas.sh \
    ${output_dir}/${subject_id}-002-1-atlas-nreg-%i \
    ${output_dir}/${subject_id}-002-1-nreg_%i \
    ${output_dir}/${subject_id}-diff.txt \
    ${output_dir}/${subject_id}-initial-atlas.hdr \
    ${output_dir}/${subject_id}-0-mask-nreg.hdr \
    1 \
    ${groupwise_atlas_command_line}
    
  # Tidy up the dofs. 
  for (( i=0; i < number_of_images; i++ ))
  do 
    ${remove} -f ${output_dir}/${subject_id}-${i}-nreg-init.dof
  done
    
  for (( i=1; i<number_of_images; i++ ))
  do
    (( j=i-1 ))
    seg_prop_command_line="${seg_prop_command_line} ${input_dir}/`basename ${input_image[$i]}` ${output_dir}/${subject_id}-${i}-average-affine.hdr ${output_dir}/${subject_id}-${j}_affine.dof ${output_dir}/${subject_id}-002-1-nreg_${i}.dof"
  done
  
  for (( i=10; i>=0; i-- ))
  do 
    if [ -f "${output_dir}/${subject_id}-002-1-atlas-nreg-${i}.img" ]
    then
      final_atlas_name=${output_dir}/${subject_id}-002-1-atlas-nreg-${i}.img
      break 
    fi 
  done
  
  ${current_dir}/GroupwiseSegProp.sh \
    ${final_atlas_name} \
    ${input_dir}/`basename ${input_region[0]}` \
    1 \
    ${input_dir}/`basename ${input_image[0]}` \
    ${output_dir}/${subject_id}-0-average-affine.hdr \
    dummy.dof \
    ${output_dir}/${subject_id}-002-1-nreg_0.dof \
    ${seg_prop_command_line}
    
  # Tidy up the dofs. 
  for (( i=0; i < number_of_images; i++ ))
  do 
    ${remove} -f ${output_dir}/${subject_id}-002-1-nreg_${i}.dof
  done
    
    


