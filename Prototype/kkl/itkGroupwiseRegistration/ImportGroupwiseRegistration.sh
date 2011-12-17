#!/bin/bash 

input_file=$1
output_dir=$2
# default dilation 4
dilation=$3
dos2unix ${input_file}
exec 3<&0

cat ${input_file} | while read each_line
do
  # copy the first image.  
  midas_fixed_image_img=`echo "${each_line}"|awk -F, '{printf $1}'`
  cp ${midas_fixed_image_img} ${output_dir}/.
  cp ${midas_fixed_image_img%.*}.hdr ${output_dir}/.
  local_fixed_image_img=${output_dir}/`basename ${midas_fixed_image_img}`
  # copy the first image region.  
  fixed_image_midas_mask=`echo "${each_line}"|awk -F, '{printf $2}'`
  cp ${fixed_image_midas_mask} ${output_dir}/.
  local_fixed_image_midas_mask=${output_dir}/`basename ${fixed_image_midas_mask}`
  local_fixed_image_mask_hdr=${output_dir}/`basename ${fixed_image_midas_mask}`_mask.hdr
  local_fixed_image_mask_img=${output_dir}/`basename ${fixed_image_midas_mask}`_mask.img
  
  makemask ${local_fixed_image_img} ${local_fixed_image_midas_mask} ${local_fixed_image_mask_img} -d ${dilation}
  
  # copy the 2nd image.  
  image_img=`echo "${each_line}"|awk -F, '{printf $3}'`
  cp ${image_img} ${output_dir}/.
  cp ${image_img%.*}.hdr ${output_dir}/.
  
  # copy the 3rd image.  
  image_img=`echo "${each_line}"|awk -F, '{printf $5}'`
  cp ${image_img} ${output_dir}/.
  cp ${image_img%.*}.hdr ${output_dir}/.
  
  # copy the 4th image.  
  image_img=`echo "${each_line}"|awk -F, '{printf $7}'`
  cp ${image_img} ${output_dir}/.
  cp ${image_img%.*}.hdr ${output_dir}/.
  
  # copy the 5th image.  
  image_img=`echo "${each_line}"|awk -F, '{printf $9}'`
  cp ${image_img} ${output_dir}/.
  cp ${image_img%.*}.hdr ${output_dir}/.
  
done
