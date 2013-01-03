#!/bin/bash
#/*================================================================================
#
#  NifTK: An image processing toolkit jointly developed by the
#              Dementia Research Centre, and the Centre For Medical Image Computing
#              at University College London.
#  
#  See:        http://dementia.ion.ucl.ac.uk/
#              http://cmic.cs.ucl.ac.uk/
#              http://www.ucl.ac.uk/
#
#  Copyright (c) UCL : See LICENSE.txt in the top level directory for details. 
#
#  Last Changed      : $LastChangedDate: 2010-08-11 08:28:23 +0100 (Wed, 11 Aug 2010) $ 
#  Revision          : $Revision: 3647 $
#  Last modified by  : $Author: mjc $
#
#  Original author   : leung@drc.ion.ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/
function Usage()
{
cat <<EOF

Working script to perform AIR vents registration and compute VBSI using automatic window selection. Please use regAIRForVentsBatch.sh. 

                       
EOF
exit 127
}

if [ $# -lt 3 ]; then
  Usage
fi

if [ "${_NIFTK_DEBUG}" == "1" ]
then 
  set -x
fi 

source _niftkCommon.sh

baseline_img=$1
baseline_mask=$2
baseline_vents_mask=$3
repeat_img=$4
repeat_mask=$5
repeat_vents_mask=$6
dof=$7
brain_dilation=$8
kn_dilation=$9
use_bbsi_reg_param=${10}
bbsi_h_param=${11}
bbsi_r_param=${12}
bbsi_threshold=${13}
bbsi_hessian=${14}
output_dir=${15}

temp_dir=`mktemp -d -q ${output_dir}/__regAIR_for_vents.XXXXXX`
function cleanup
{
  echo "Cleaning up..."
  rm -rf ${temp_dir}
  exit
}
trap "cleanup" EXIT SIGINT SIGTERM SIGKILL 

baseline_image_basename=`basename ${baseline_img}`
repeat_image_basename=`basename ${repeat_img}`
log_file=${output_dir}/${baseline_image_basename}_${repeat_image_basename}_vbsi.csv
repeat_id=`echo ${repeat_image_basename} | awk -F- '{printf $1}'`
output_brain_air=${output_dir}/${baseline_image_basename}_${repeat_image_basename}_brain.air
output_air=${output_dir}/${baseline_image_basename}_${repeat_image_basename}.air

if [ ${brain_dilation} -gt 0 ] 
then 
  brain_dilation_flag="-d ${brain_dilation}"
fi 

# Baseline and repeat brain masks. 
makemask ${baseline_img}.img ${baseline_mask} ${temp_dir}/baseline_brain_mask ${brain_dilation_flag}
makemask ${repeat_img}.img ${repeat_mask} ${temp_dir}/repeat_brain_mask ${brain_dilation_flag}

# Only baseline vents mask. 
makemask ${baseline_img}.img ${baseline_vents_mask} ${temp_dir}/baseline_local_mask -d 2

if [ "${use_bbsi_reg_param}" == "no" ]
then 
  # Brain-to-brain registration. 
  ${AIR_BIN}/alignlinear ${baseline_img}.img ${repeat_img}.img \
                        ${output_brain_air} -m ${dof} -h 50 \
                        -e1 ${temp_dir}/baseline_brain_mask.img -e2 ${temp_dir}/repeat_brain_mask.img \
                        -x 1 -v -r 100 -c 0.00000001 -s 81 1 3 -g ${temp_dir}/brain.init y \
                        -p1 1 -p2 1 -b1 1.875 1.875 3.000 -b2 1.875 1.875 3.000
  
  ${AIR_BIN}/alignlinear ${baseline_img}.img ${repeat_img}.img \
                        ${output_brain_air} -m ${dof}  -h 50 \
                        -e1 ${temp_dir}/baseline_brain_mask.img -e2 ${temp_dir}/repeat_brain_mask.img \
                        -x 1 -v -r 100 -c 0.00000001 -s 81 1 3 -t1 0 -t2 0 -f ${temp_dir}/brain.init \
                        -p1 1 -p2 1 
else
  t1=`imginfo ${baseline_img}.img -tanz 0.2`
  t2=`imginfo ${repeat_img}.img -tanz 0.2`
  threshold_flag=""
  bbsi_hessian_flag=""
  
  if [ "${bbsi_threshold}" == "yes" ]
  then 
    threshold_flag="-t1 ${t1} -t2 ${t2}"
  fi 
  
  if [ "${bbsi_hessian}" == "yes" ] 
  then 
    bbsi_hessian_flag="-q"
  fi 

  # Brain-to-brain registration. 
  ${AIR_BIN}/alignlinear ${baseline_img}.img ${repeat_img}.img \
                        ${output_brain_air} -m ${dof}  \
                        -e1 ${temp_dir}/baseline_brain_mask.img -e2 ${temp_dir}/repeat_brain_mask.img \
                        -x 1 -v -c 0.00000001 -s 81 1 3 -g ${temp_dir}/brain.init y \
                        -p1 1 -p2 1 -b1 1.875 1.875 3.000 -b2 1.875 1.875 3.000 -h ${bbsi_h_param} -r ${bbsi_r_param} ${bbsi_hessian_flag} \
                        ${threshold_flag}
  
  ${AIR_BIN}/alignlinear ${baseline_img}.img ${repeat_img}.img \
                        ${output_brain_air} -m ${dof} \
                        -e1 ${temp_dir}/baseline_brain_mask.img -e2 ${temp_dir}/repeat_brain_mask.img \
                        -x 1 -v -c 0.00000001 -s 2 1 2 -f ${temp_dir}/brain.init \
                        -p1 1 -p2 1 -h ${bbsi_h_param} -r ${bbsi_r_param} ${bbsi_hessian_flag} \
                        ${threshold_flag}
fi 
                       
                       
# Reslice the repeat image using the brain-to-brain regsitration result.                        
${AIR_BIN}/reslice ${output_brain_air} ${temp_dir}/90001-007-1 -k -n 10
extend_header ${temp_dir}/90001-007-1.img ${repeat_img}.img ${temp_dir} 7 1 10001
rm -f ${temp_dir}/90001-007-1.*
                       
# Reslice the repeat brain region and repeat vents region using the brain-to-brain registration result. 
regslice ${output_brain_air} ${repeat_mask} ${temp_dir}/brr_brain_region 7 -s 10001 -i 2
regslice ${output_brain_air} ${repeat_vents_mask} ${temp_dir}/brr_local_region 7 -s 10001 -i 2

# Registered repeat brain region mask. 
makemask ${temp_dir}/10001-007-1.img ${temp_dir}/brr_brain_region ${temp_dir}/brr_brain_mask

# Vents-to-vents registration. 
${AIR_BIN}/alignlinear ${temp_dir}/10001-007-1.img ${baseline_img}.img \
                       ${temp_dir}/invlocal.air -m 6 -h 50 \
                       -e1 ${temp_dir}/brr_brain_mask.img -e2 ${temp_dir}/baseline_local_mask.img \
                       -t1 0 -t2 0 -x 1 -v -r 100 -c 0.0000001 -s 1 1 3 
                       
# Reslice the repeat image.                     
${AIR_BIN}/invert_air ${temp_dir}/invlocal.air ${output_air} y
${AIR_BIN}/reslice ${output_air} ${temp_dir}/90002-007-1 -k -n 10
extend_header ${temp_dir}/90002-007-1.img ${temp_dir}/10001-007-1.img ${temp_dir} 7 1 10002
extend_header ${temp_dir}/90002-007-1.img ${temp_dir}/10001-007-1.img ${output_dir} 17 1 ${repeat_id}
rm -f ${temp_dir}/90002-007-1.*

# Reslice the repeat brain and vents regions. 
regslice ${output_air} ${temp_dir}/brr_local_region ${temp_dir}/lrr_local_region 7 -s 10002 -i 2
regslice ${output_air} ${temp_dir}/brr_brain_region ${temp_dir}/lrr_brain_region 7 -s 10002 -i 2
regslice ${output_air} ${temp_dir}/brr_local_region ${output_dir} 17 -s ${repeat_id} -c -i 2 -q "vents locreg"
sleep 2
regslice ${output_air} ${temp_dir}/brr_brain_region ${output_dir} 17 -s ${repeat_id} -c -i 2 -q "brain locreg"

# Calculate the VBSI. 
makemask ${baseline_img}.img ${baseline_mask} ${temp_dir}/baseline_brain_mask_bsi.img 
makemask ${baseline_img}.img ${baseline_vents_mask} ${temp_dir}/baseline_local_mask_bsi.img 
makemask ${temp_dir}/10002-007-1.img ${temp_dir}/lrr_brain_region ${temp_dir}/lrr_brain_region.img
makemask ${temp_dir}/10002-007-1.img ${temp_dir}/brr_local_region ${temp_dir}/brr_local_region.img
niftkKMeansWindowWithLinearRegressionNormalisationBSI \
  ${baseline_img}.img ${temp_dir}/baseline_brain_mask_bsi.img \
  ${temp_dir}/10002-007-1.img ${temp_dir}/lrr_brain_region.img \
  ${baseline_img}.img ${temp_dir}/baseline_local_mask_bsi.img \
  ${temp_dir}/10002-007-1.img ${temp_dir}/brr_local_region.img \
  1 1 ${kn_dilation} -1 -1 ${temp_dir}/baseline_image_seg.hdr ${temp_dir}/repeat_image_seg.hdr ${temp_dir}/repeat_image_normalised.hdr > ${log_file}













