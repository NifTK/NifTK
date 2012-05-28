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
#  Last Changed      : $LastChangedDate: 2011-08-17 11:43:34 +0100 (Wed, 17 Aug 2011) $ 
#  Revision          : $Revision: 7120 $
#  Last modified by  : $Author: kkl $
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

This script is the working script which is called Hippo-MAPS.sh to perform automated hippo segmentation. Please use Hippo-MAPS-batch.sh. 

EOF
exit 127
}

# Check args
if [ $# -lt 3 ]; then
  Usage
fi



# Debug
#set -x

# No undefined varaiable. 
set -u 

#
# Script to delineate hippo using a template library 
# See A comparison of methods for the automated calculation of volumes and atrophy rates in the hippocampus
#     J. Barnes et al, NeuroImage, 2008. 
#
# Added FFD and STAPLE to improve the segmentation. 
# 


if [ ! -n "${MIDAS_BIN}" ]
then 
  echo "Please define MIDAS_BIN to midas bin directory in your shelll environment."
  exit
fi 
if [ ! -n "${FFITK_BIN}" ]
then
  echo "Please set FFITK_BIN to the IRTK bin directory in your shelll environment."
  exit
fi 
if [ ! -n "${MIDAS_FFD}" ]
then
  echo "Please set MIDAS_FFD to the midas ffd script directory in your shelll environment."
  exit
fi 
if [ ! -n "${FSLDIR}" ]
then 
  echo "Please set FSLDIR to the FSL directory in your shelll environment."
  exit
fi 
if [ ! -n "${F3D_BIN}" ]
then 
  echo "Please set F3D_BIN to the Nifti-Reg directory in your shelll environment."
  exit
fi 

# Set up all the essential programs. 
export reg_template_loc=${MIDAS_BIN}/reg-template-loc
export EVALUATION=${FFITK_BIN}/evaluation
export ffdevaluate=${MIDAS_FFD}/ffdevaluate.sh
export ffdnreg=${MIDAS_FFD}/ffdnreg.sh
export ffdareg=${MIDAS_FFD}/ffdareg.sh
export ffdtransformation=${MIDAS_FFD}/ffdtransformation.sh
export ffdroitransformation=${MIDAS_FFD}/ffdroitransformation.sh
export ffdsubdivide=${FFITK_BIN}/ffdsubdivide
export threshold=niftkThreshold
export crlSTAPLE=crlSTAPLE
export crlExtractSmallerImageFromImage=crlExtractSmallerImageFromImage
export crlMeanFieldMRF=crlMeanFieldMRF
export thresholdProb=niftkShiftProb
export nifti_reg=${F3D_BIN}/reg_f3d
export nifti_resample=${F3D_BIN}/reg_resample
export abs_filter=niftkAbsImageFilter
export convert=niftkConvertImage

# Bending energy for F3D. default: 0.01. 
f3d_bending_energy=0.01

# Dilation to create a ROI used in the F3D registration. default: 16. 
nreg_hippo_dilation=4

# Staple confidence level for determining a foreground voxel. 
staple_confidence=0.99999

# Source the common function. 
source MAPS-common.sh

#
# Match the hippo and generate the cross-correlation with all the images in the library. 
#
function hippo-match()
{
  echo "number of arguments=$#"

  local watjo_image=$1
  local watjo_brain_region=$2
  local subject_image=$3
  local subject_brain_region=$4
  local output_brain_image=$5
  local output_brain_region=$6
  local output_brain_series_number=$7
  local output_brain_air=$8
  local watjo_hippo_left_region=$9
  local output_left_match_dir=${10}
  local output_study_id=${11}
  local output_left_series_number=${12}
  local output_left_echo_number=${13}
  local watjo_hippo_right_region=${14}
  local output_right_match_dir=${15}
  local output_right_series_number=${16}
  local output_right_echo_number=${17}
  local left_hippo_template_library=${18}
  local output_left_corr=${19}
  local right_hippo_template_library=${20}
  local output_right_corr=${21}
  local flipx=${22}
  local leave_one_out=${23}
  local hippo_template_library_original=${24}

  if [ 1 = 1 ] 
  then
  brain_to_brain_registration ${watjo_image} ${watjo_brain_region} ${subject_image} ${subject_brain_region} \
    ${output_brain_image} ${output_brain_region} ${output_brain_series_number} ${output_brain_air}

  # Left hippo
  ${reg_template_loc} ${watjo_image} ${output_brain_image} ${watjo_brain_region} ${output_brain_region} \
                      ${watjo_hippo_left_region} dummy \
                      ${output_left_match_dir} ${output_left_match_dir} ${output_left_match_dir} ${output_left_match_dir} \
                      ${output_study_id} ${output_left_series_number} ${output_left_echo_number} 0.5 0.8 \
                      no ${output_study_id}_xor.roi 4 1 1 no 

  # Right hippo
  ${reg_template_loc} ${watjo_image} ${output_brain_image} ${watjo_brain_region} ${output_brain_region} \
                      ${watjo_hippo_right_region} dummy \
                      ${output_right_match_dir} ${output_right_match_dir} ${output_right_match_dir} ${output_right_match_dir} \
                      ${output_study_id} ${output_right_series_number} ${output_right_echo_number} 0.5 0.8 \
                      no ${output_study_id}_xor.roi 4 1 1 no 
                      
  fi                      
                      
  local output_left_image=${output_left_match_dir}/${output_study_id}-${output_left_series_number}-${output_left_echo_number}.img
  local output_right_image=${output_right_match_dir}/${output_study_id}-${output_right_series_number}-${output_right_echo_number}.img
                      
  rm -f ${output_right_corr} 
  exec <  ${right_hippo_template_library} 
  while read each_line 
  do
    local template_right_image=${hippo_template_library_dir}/`echo ${each_line}|awk -F, '{print $1}'`
    local template_right_region=${hippo_template_library_dir}/`echo ${each_line}|awk -F, '{print $2}'`
    
    if [ "${leave_one_out}" == "yes" ]
    then 
      template_image_basename=`basename ${template_right_image}`
      original_image_name=`grep ${template_image_basename} ${hippo_template_library_original} |awk -F, '{printf $8}'`
      original_image_basename=`basename ${original_image_name}`
      subject_image_basename=`basename ${subject_image}`
      if [ "${original_image_basename}" == "${subject_image_basename}" ]
      then 
        echo "Skipping same image: ${template_right_image}, ${original_image_name}, ${subject_image}..." 
        continue
      fi 
    fi 
    
    local right_corr=`${ffdevaluate} ${template_right_image} ${output_right_image} -troi ${template_right_region} | grep Crosscorrelation| awk '{print $2}'`
    
    echo "right_corr=${right_corr}"
    echo "${right_corr},${template_right_image},${template_right_region},${flipx}" >> ${output_right_corr}
  done
  
  rm -f ${output_left_corr}
  exec <  ${left_hippo_template_library} 
  while read each_line 
  do
    local template_left_image=${hippo_template_library_dir}/`echo ${each_line}|awk -F, '{print $1}'`
    local template_left_region=${hippo_template_library_dir}/`echo ${each_line}|awk -F, '{print $2}'`
    
    if [ "${leave_one_out}" == "yes" ]
    then 
      template_image_basename=`basename ${template_left_image}`
      original_image_name=`grep ${template_image_basename} ${hippo_template_library_original} |awk -F, '{printf $8}'`
      original_image_basename=`basename ${original_image_name}`
      subject_image_basename=`basename ${subject_image}`
      if [ "${original_image_basename}" == "${subject_image_basename}" ]
      then 
        echo "Skipping same image: ${template_left_image}, ${original_image_name}, ${subject_image}..." 
        continue
      fi 
    fi 
    
    local left_corr=`${ffdevaluate} ${template_left_image} ${output_left_image} -troi ${template_left_region} | grep Crosscorrelation| awk '{print $2}'`
    
    echo "left_corr=${left_corr}"
    echo "${left_corr},${template_left_image},${template_left_region},${flipx}" >> ${output_left_corr}
  done
  
  rm -f ${output_brain_image} ${output_brain_image%.img}.hdr
  rm -f ${output_left_image} ${output_left_image%.img}.hdr
  rm -f ${output_right_image} ${output_right_image%.img}.hdr

  echo "Hippo-match done"
}


#
# Delineate one hippo by stapling multiple segmentations. 
#
function hippo-delineation-using-staple()
{
  local left_or_right=${1}
  local output_left_corr=${2}
  local output_right_corr_flipx=${3}
  local subject_image=${4}
  local subject_brain_region=${5}
  local hippo_template_library_dir=${6}
  local hippo_template_library_original=${7}
  local output_left_dir=${8}
  local mrf_weighting=${9}
  local do_threshold=${10}
  local mean_brain_intensity=${11}
  local remove_dir=${12}
  local only_staple=${13}
  
  # Select the best match ones from the library - including flipped images. 
  local temp_left_corr_file=`mktemp /usr/tmp/left_corr.XXXXXX`
  cat ${output_right_corr_flipx} ${output_left_corr} | sort -nr | head -n ${staple_count}  > ${temp_left_corr_file}
  cat ${temp_left_corr_file}
  
  local staple_command_line_nreg_thresholded=""
  local result_dir=""
  exec < ${temp_left_corr_file} 
  while read each_line
  do 
    # A bunch of messy code to find out the template image. 
    echo ${each_line}
    local left_image=`echo ${each_line} | awk -F, '{printf $2}'`
    local left_image_basename=`basename ${left_image}`
    echo ${left_image_basename}
    flipx=`echo ${each_line} | awk -F, '{printf $4}'`
    
    if [[ "${flipx}" = "flipx" ]]
    then
      local old_subject_image=${subject_image}
      local old_subject_brain_region=${subject_brain_region}
      local temp_flipx_dir=`mktemp -d /usr/tmp/_flipx.XXXXXX`
      local subject_image_flipx=${temp_flipx_dir}/`basename ${subject_image}`
      anchange ${subject_image} ${subject_image_flipx} -flipx
      local dims=`imginfo ${subject_image_flipx} -dims | awk '{printf $1" "$2" "$3}'`
      local subject_brain_region_flipx=${temp_flipx_dir}/`basename ${subject_brain_region}`
      regchange ${subject_brain_region} ${subject_brain_region_flipx} ${dims} -flipx
      local subject_image=${subject_image_flipx}
      local subject_brain_region=${subject_brain_region_flipx}
    fi 
    
    local template_left_image=""
    exec 6<&0
    exec < ${hippo_template_library_original}
    while read line 
    do
      local each_image=`echo ${line}|awk -F, '{print $2}'`
      if [[ "${left_or_right}" = "left" && "${flipx}" = "flipx" ]] || [[ "${left_or_right}" = "right" && "${flipx}" = "noflipx" ]]
      then
        local each_image=`echo ${line}|awk -F, '{print $3}'`
      fi 
      
      if [ ${each_image} = ${left_image_basename} ] 
      then 
        local template_left_image=${hippo_template_library_dir}/`echo ${line}|awk -F, '{print $8}'`
        local template_left_brain_region=${hippo_template_library_dir}/`echo ${line}|awk -F, '{print $9}'`
        local template_left_hippo_region=${hippo_template_library_dir}/`echo ${line}|awk -F, '{print $10}'`
        if [[ "${left_or_right}" = "left" && "${flipx}" = "flipx" ]] || [[ "${left_or_right}" = "right" && "${flipx}" = "noflipx" ]]
        then
          local template_left_hippo_region=${hippo_template_library_dir}/`echo ${line}|awk -F, '{print $11}'`
        fi
      fi 
    done
    exec 0<&6 6<&-
    if [ ! -f "${template_left_image}" ]
    then
      echo "Cannot locate original image for ${left_image}"
      exit
    fi
    
    echo ${template_left_image},${template_left_brain_region},${template_left_hippo_region}
    local template_prefix=`basename ${template_left_image} | awk -F. '{print $1}'`
    
    # Create the directory structure
    local output_left_delineate_dir=${output_left_dir}/delineate/${output_study_id}-${template_prefix}
    if [ "${flipx}" = "flipx" ]
    then
      local output_left_delineate_dir=${output_left_dir}/delineate/${output_study_id}-${template_prefix}-flipx
    fi
    
    mkdir -p ${output_left_delineate_dir} 
    # For AIR. 
    chmod g+w ${output_left_delineate_dir} 
    
    local output_areg_template_brain_image=${output_left_delineate_dir}/${output_study_id}-${output_areg_template_brain_series_number}-${output_left_echo_number}.img
    local output_areg_template_brain_region=${output_left_delineate_dir}/`basename ${subject_brain_region}`
    local output_areg_template_air=${output_left_delineate_dir}/${output_study_id}.air
    local output_left_image=${output_left_delineate_dir}/${output_study_id}-${output_left_series_number}-${output_left_echo_number}.img
    local output_areg_hippo_region=${output_left_delineate_dir}/`basename ${subject_brain_region}`-areg
    local output_local_areg_hippo_region=${output_left_delineate_dir}/`basename ${subject_brain_region}`-areg-local
    local output_nreg_hippo_region=${output_left_delineate_dir}/`basename ${subject_brain_region}`-nreg
    local output_nreg_template_hippo_image=${output_left_delineate_dir}/${output_study_id}-${output_left_series_number}-${output_left_echo_number}-hippo.img
    local output_nreg_template_hippo_dof_16=${output_left_delineate_dir}/${output_study_id}-${output_left_series_number}-${output_left_echo_number}-hippo_16.dof
    local output_nreg_template_hippo_dof_8=${output_left_delineate_dir}/${output_study_id}-${output_left_series_number}-${output_left_echo_number}-hippo_8.dof
    local output_nreg_template_hippo_dof_4=${output_left_delineate_dir}/${output_study_id}-${output_left_series_number}-${output_left_echo_number}-hippo_4.dof
    local output_areg_mm_hippo_region=${output_left_delineate_dir}/`basename ${subject_brain_region}`-areg-mm
    local output_nreg_mm_hippo_region=${output_left_delineate_dir}/`basename ${subject_brain_region}`-nreg-mm
    local output_nreg_thresholded_hippo_region=${output_left_delineate_dir}/`basename ${subject_brain_region}`-nreg-thresholded
  
    # Delineate the hippo.   
    hippo_delineation ${subject_image} ${subject_brain_region} ${template_left_image} ${template_left_brain_region} \
                      ${output_areg_template_brain_image} ${output_areg_template_brain_region} \
                      ${output_areg_template_brain_series_number} ${output_areg_template_air} \
                      ${template_left_hippo_region} ${output_areg_hippo_region} \
                      ${output_left_delineate_dir} ${output_study_id} ${output_left_series_number} ${output_left_echo_number}  \
                      ${output_local_areg_hippo_region} ${output_nreg_hippo_region} ${output_nreg_template_hippo_image} \
                      ${output_nreg_template_hippo_dof_16} ${output_nreg_template_hippo_dof_8} ${output_nreg_template_hippo_dof_4} \
                      ${output_areg_mm_hippo_region} ${output_nreg_mm_hippo_region} \
                      ${output_nreg_thresholded_hippo_region} ${do_threshold} ${mean_brain_intensity} ${only_staple}
    
    makemask ${subject_image} ${output_nreg_thresholded_hippo_region} ${output_nreg_thresholded_hippo_region}.img -val 1
    local staple_command_line_nreg_thresholded="${staple_command_line_nreg_thresholded} ${output_nreg_thresholded_hippo_region}.img"
    local result_dir="${result_dir} ${output_left_delineate_dir}" 
    
    if [ "${flipx}" = "flipx" ]
    then
      rm -rf ${temp_flipx_dir}
      local subject_image=${old_subject_image}
      local subject_brain_region=${old_subject_brain_region}
      anchange ${output_nreg_thresholded_hippo_region}.img ${output_nreg_thresholded_hippo_region}.img -flipx
    fi  
    
  done
  rm -f ${temp_left_corr_file}
  
  local output_hippo_areg_region=${output_left_dir}/delineate/${output_study_id}-${output_areg_template_brain_series_number}-${output_left_echo_number}-hippo-areg
  local output_hippo_staple_nreg_thresholded_region=${output_left_dir}/delineate/${output_study_id}-${output_areg_template_brain_series_number}-${output_left_echo_number}-hippo-staple-nreg-thresholded
  local output_hippo_staple_nreg_thresholded_staple=${output_hippo_staple_nreg_thresholded_region}-staple.img
  local output_hippo_staple_nreg_thresholded=${output_hippo_staple_nreg_thresholded_region}.img
  
  local output_hippo_staple_weights=${output_left_dir}/delineate/${output_study_id}-weights.nrrd
  local output_hippo_staple_foregroundprob=${output_left_dir}/delineate/${output_study_id}-foreground-prob.img
  local output_hippo_staple_mf_weights=${output_left_dir}/delineate/${output_study_id}-mfweights.nrrd
  local output_hippo_staple_mf_region=${output_left_dir}/delineate/${output_study_id}-staple-mrf-${staple_count}-${do_threshold}
  local output_hippo_staple_mf_seg=${output_hippo_staple_mf_region}.img
  local output_hippo_staple_weights_thresholded=${output_left_dir}/delineate/${output_study_id}-weights-thresholded.nrrd
  
  ${crlSTAPLE} --outputImage ${output_hippo_staple_weights} ${staple_command_line_nreg_thresholded} 
  ${crlExtractSmallerImageFromImage} -i ${output_hippo_staple_weights} -o ${output_hippo_staple_nreg_thresholded} -l 1 -m 2 -a 3
  ${threshold} -i ${output_hippo_staple_nreg_thresholded} -o ${output_hippo_staple_nreg_thresholded} -u 2 -l ${staple_confidence} -in 255 -out 0
  ${convert} -i ${output_hippo_staple_nreg_thresholded} -o ${output_hippo_staple_nreg_thresholded} -ot short
  makeroi -img ${output_hippo_staple_nreg_thresholded} -out ${output_hippo_staple_nreg_thresholded_region} -alt 128
  
  # reset the probability to 0.5 if it is less than the staple_confidence, in order to allow the MRF not too smooth the segmentation. 
  ${thresholdProb} ${output_hippo_staple_weights} ${staple_confidence} 0.5 ${output_hippo_staple_weights_thresholded}
  
  ${crlMeanFieldMRF} ${output_hippo_staple_weights_thresholded} automatic 0.00001 ${mrf_weighting} 5 ${output_hippo_staple_mf_weights}
  ${crlExtractSmallerImageFromImage} -i ${output_hippo_staple_mf_weights} -o ${output_hippo_staple_mf_seg} -l 1 -m 2 -a 3
  ${threshold} -i ${output_hippo_staple_mf_seg} -o ${output_hippo_staple_mf_seg} -u 2 -l ${staple_confidence} -in 255 -out 0
  ${convert} -i ${output_hippo_staple_mf_seg} -o ${output_hippo_staple_mf_seg} -ot short
  makeroi -img ${output_hippo_staple_mf_seg} -out ${output_hippo_staple_mf_region} -alt 128
    
  local dims=`imginfo ${subject_image} -dims | awk '{printf $1" "$2" "$3}'`
  # These flipping and re-orienting are only for ADNI scans. 
  regchange ${output_hippo_staple_nreg_thresholded_region} ${output_hippo_staple_nreg_thresholded_region} ${dims} -flipx
  regchange ${output_hippo_staple_nreg_thresholded_region} ${output_hippo_staple_nreg_thresholded_region}_${staple_count} ${dims} -orient cor sag
  regchange ${output_hippo_staple_mf_region} ${output_hippo_staple_mf_region}_${staple_count}_${mrf_weighting} ${dims} -flipx
  regchange ${output_hippo_staple_mf_region}_${staple_count}_${mrf_weighting} ${output_hippo_staple_mf_region}_${staple_count}_${mrf_weighting} ${dims} -orient cor sag
  
  local image_info=`imginfo ${subject_image} -info`
  local name=`echo ${image_info}|awk '{print $5}'`
  local study_id=`echo ${image_info}|awk '{print $1}'`
  local series_no=`echo ${image_info}|awk '{print $2}'`
  local scan_date=`imginfo ${subject_image} -info -datefmt|awk '{print $4 " " $5 " " $6 " " $7}'`
  local output_region_name=. 
  local quality=MAPS
  local structure="LeftHippocampus"
  local segmentor=Hplauto
  if [ "${left_or_right}" == "right" ] 
  then 
    local structure="RightHippocampus"
    local segmentor=Hprauto
  fi 
  
  regchange ${output_hippo_staple_mf_region} ${output_left_dir}/delineate/. ${dims} -study ${study_id} -series ${series_no} -name "${name}" -segmentor ${segmentor} -structure ${structure} -quality ${quality} -acq "${scan_date}" -c
    
  rm -f ${output_hippo_staple_nreg_thresholded_staple} ${output_hippo_staple_nreg_thresholded_staple%.img}.hdr ${output_hippo_staple_nreg_thresholded} ${output_hippo_staple_nreg_thresholded%.img}.hdr
  rm -f ${output_hippo_staple_weights} ${output_hippo_staple_mf_weights} ${output_hippo_staple_mf_seg} ${output_hippo_staple_mf_seg%.img}.hdr
  
  rm -f ${staple_command_line_nreg_thresholded}
  rm -f ${output_areg_mm_hippo_region}.img ${output_areg_mm_hippo_region}.hdr
  
  # remove the staple weights. 
  rm -f ${output_hippo_staple_weights} ${output_hippo_staple_weights_thresholded}
  # remove the temp directories. 
  if [ "${remove_dir}" == "yes" ] && [ -f ${output_hippo_staple_mf_region} ]
  then 
    rm -rf ${result_dir}
  fi 
}




echo "Starting... on ${HOSTNAME}"

hippo_template_library_index=$1
subject_image=$2
subject_brain_region=$3
output_study_id=$4
output_left_dir=$5
output_right_dir=$6
# staple_count is now iterating from staple_count_start to staple_count_end. 
staple_count=$7     
mrf_weighting=$8
do_threshold=$9
mean_brain_intensity=${10}
remove_dir=${11}
leave_one_out=${12}
only_staple=${13}
process_left=${14}
process_right=${15}
staple_count_start=${16}
staple_count_end=${17}

output_areg_template_brain_series_number=400
output_left_series_number=665
output_right_series_number=666
output_left_echo_number=1
output_right_echo_number=1

if [ ! -f "${hippo_template_library_index}" ]
then 
  hippo_template_library_dir=${hippo_template_library_index}
  left_hippo_template_library=${hippo_template_library_dir}/left-hippo-template-library.csv
  right_hippo_template_library=${hippo_template_library_dir}/right-hippo-template-library.csv
  hippo_template_library_original=${hippo_template_library_dir}/standard-space-template-library.csv
  watjo_image=${hippo_template_library_dir}/30063-333-1.img
  watjo_brain_region=${hippo_template_library_dir}/Rhi_30063_975928052
  watjo_hippo_left_region=${hippo_template_library_dir}/Ash_30063_1098371494
  watjo_hippo_right_region=${hippo_template_library_dir}/Ash_30063_1098357734
else
  hippo_template_library_dir=`dirname ${hippo_template_library_index}`
  left_hippo_template_library=${hippo_template_library_dir}/`cat ${hippo_template_library_index} | awk -F, '{printf $1}'`
  right_hippo_template_library=${hippo_template_library_dir}/`cat ${hippo_template_library_index} | awk -F, '{printf $2}'`
  hippo_template_library_original=${hippo_template_library_dir}/`cat ${hippo_template_library_index} | awk -F, '{printf $3}'`
  watjo_image=${hippo_template_library_dir}/`cat ${hippo_template_library_index} | awk -F, '{printf $4}'`
  watjo_brain_region=${hippo_template_library_dir}/`cat ${hippo_template_library_index} | awk -F, '{printf $5}'`
  watjo_hippo_left_region=${hippo_template_library_dir}/`cat ${hippo_template_library_index} | awk -F, '{printf $6}'`
  watjo_hippo_right_region=${hippo_template_library_dir}/`cat ${hippo_template_library_index} | awk -F, '{printf $7}'`
fi 

# Create the directory structure
output_left_match_dir=${output_left_dir}/match
output_right_match_dir=${output_right_dir}/match
mkdir -p ${output_left_match_dir} ${output_right_match_dir} 
# For AIR. 
chmod g+w ${output_left_match_dir} ${output_right_match_dir} 

output_brain_image=${output_left_match_dir}/${output_study_id}-${output_areg_template_brain_series_number}-${output_left_echo_number}.img
output_brain_region=${output_left_match_dir}/`basename ${subject_brain_region}`
output_brain_air=${output_left_match_dir}/${output_study_id}.air

output_left_corr=${output_left_match_dir}/${output_study_id}-corr.txt
output_right_corr=${output_right_match_dir}/${output_study_id}-corr.txt

if [ "${only_staple}" == "no" ]
then 
  hippo-match ${watjo_image} ${watjo_brain_region} ${subject_image} ${subject_brain_region} \
              ${output_brain_image} ${output_brain_region} ${output_areg_template_brain_series_number} ${output_brain_air}  \
              ${watjo_hippo_left_region} ${output_left_match_dir} \
              ${output_study_id} ${output_left_series_number} ${output_left_echo_number} \
              ${watjo_hippo_right_region} ${output_right_match_dir} \
              ${output_right_series_number} ${output_right_echo_number} \
              ${left_hippo_template_library} ${output_left_corr} \
              ${right_hippo_template_library} ${output_right_corr} noflipx ${leave_one_out} ${hippo_template_library_original}
fi             
            
rm -f ${output_brain_image} ${output_brain_image%.img}.hdr
            
output_left_match_flipx_dir=${output_left_dir}/match-flipx
output_right_match_flipx_dir=${output_right_dir}/match-flipx
mkdir -p ${output_left_match_flipx_dir} ${output_right_match_flipx_dir} 
# For AIR. 
chmod g+w ${output_left_match_flipx_dir} ${output_right_match_flipx_dir} 

temp_flipx_dir=`mktemp -d /usr/tmp/_flipx.XXXXXX`

subject_image_flipx=${temp_flipx_dir}/`basename ${subject_image}`
anchange ${subject_image} ${subject_image_flipx} -flipx
dims=`imginfo ${subject_image_flipx} -dims | awk '{printf $1" "$2" "$3}'`
subject_brain_region_flipx=${temp_flipx_dir}/`basename ${subject_brain_region}`
regchange ${subject_brain_region} ${subject_brain_region_flipx} ${dims} -flipx

output_brain_image_flipx=${output_left_match_flipx_dir}/${output_study_id}-${output_areg_template_brain_series_number}-${output_left_echo_number}.img
output_brain_region_flipx=${output_left_match_flipx_dir}/`basename ${subject_brain_region}`
output_brain_air_flipx=${output_left_match_flipx_dir}/${output_study_id}.air
output_left_corr_flipx=${output_left_match_flipx_dir}/${output_study_id}-corr.txt
output_right_corr_flipx=${output_right_match_flipx_dir}/${output_study_id}-corr.txt

if [ "${only_staple}" == "no" ]
then 
  hippo-match ${watjo_image} ${watjo_brain_region} ${subject_image_flipx} ${subject_brain_region_flipx} \
              ${output_brain_image_flipx} ${output_brain_region_flipx} ${output_areg_template_brain_series_number} ${output_brain_air_flipx}  \
              ${watjo_hippo_left_region} ${output_left_match_flipx_dir} \
              ${output_study_id} ${output_left_series_number} ${output_left_echo_number} \
              ${watjo_hippo_right_region} ${output_right_match_flipx_dir} \
              ${output_right_series_number} ${output_right_echo_number} \
              ${left_hippo_template_library} ${output_left_corr_flipx} \
              ${right_hippo_template_library} ${output_right_corr_flipx} flipx ${leave_one_out} ${hippo_template_library_original}
fi             
            
rm -f ${output_brain_image_flipx} ${output_brain_image_flipx%.img}.hdr
rm -rf ${temp_flipx_dir}            

for (( staple_count = staple_count_start; staple_count <= staple_count_end; staple_count++ ))
do 
  if [ "${process_left}" == "yes" ]
  then 
    hippo-delineation-using-staple left ${output_left_corr} ${output_right_corr_flipx} ${subject_image} \
          ${subject_brain_region} ${hippo_template_library_dir} ${hippo_template_library_original} ${output_left_dir} ${mrf_weighting} \
          ${do_threshold} ${mean_brain_intensity} ${remove_dir} ${only_staple}
  fi 
          
  if [ "${process_right}" == "yes" ]
  then 
    hippo-delineation-using-staple right ${output_right_corr} ${output_left_corr_flipx} ${subject_image} \
          ${subject_brain_region} ${hippo_template_library_dir} ${hippo_template_library_original} ${output_right_dir} ${mrf_weighting} \
          ${do_threshold} ${mean_brain_intensity} ${remove_dir} ${only_staple}
  fi         
done        




