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
#  Last Changed      : $LastChangedDate: 2011-09-29 13:39:55 +0100 (Thu, 29 Sep 2011) $ 
#  Revision          : $Revision: 7402 $
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

This script is the working script which is called Brain-MAPS-batch.sh to perform automated brain segmentation. Please use Brain-MAPS-batch.sh. 
  
EOF
exit 127
}

if [ $# -lt 3 ]
then 
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
export seg_stat=niftkSegmentationStatistics
export combine=niftkCombineSegmentations

# Bending energy for F3D. default: 0.01. 
f3d_bending_energy=0.01

# Dilation to create a ROI used in the F3D registration. default: 16. 
nreg_hippo_dilation=4

# Staple confidence level for determining a foreground voxel. 
staple_confidence=0.99999

# Source the common function. 
source MAPS-common.sh
source MAPS-brain-to-brain-registration-without-repeat-mask.sh

#
# Match the hippo and generate the cross-correlation with all the images in the library. 
#
function brain-match()
{
  echo "number of arguments=$#"
  minarg=10
  if [ $# != $minarg ]
  then
    echo "brain match requires $minarg arguments"
    exit
  fi

  local watjo_image=$1
  local watjo_brain_region=$2
  local subject_image=$3
  local output_brain_image=$4
  local output_brain_series_number=$5
  local output_brain_air=$6
  local left_hippo_template_library=${7}
  local output_left_corr=${8}
  local areg=${9}
  local leaveoneout=${10}

  local subject_image_basename=`basename ${subject_image}`
  local subject_image_id=`echo ${subject_image_basename} | awk -F- '{printf $1}'`
  local subject_patient_id=`imginfo ${subject_image} -info | awk '{printf $5}'`
  
  if [ "${areg}" == "air" ]
  then 
    brain_to_brain_registration_without_repeat_mask ${watjo_image} ${watjo_brain_region} ${subject_image}  \
      ${output_brain_image} ${output_brain_series_number} ${output_brain_air} "no" 
  else
    brain_to_brain_registration_without_repeat_mask_using_irtk ${watjo_image} ${watjo_brain_region} ${subject_image}   \
        ${output_brain_image} ${output_brain_series_number} ${output_brain_air}.dof "no" "${init_9dof}"
  fi 
  
  rm -f ${output_left_corr}
  exec <  ${left_hippo_template_library} 
  while read each_line 
  do
    local template_left_image=${hippo_template_library_dir}/`echo ${each_line}|awk -F, '{print $1}'`
    local template_left_region=${hippo_template_library_dir}/`echo ${each_line}|awk -F, '{print $2}'`
    local template_left_image_basename=`basename ${template_left_image}`
    local template_left_image_id=`echo ${template_left_image_basename} | awk -F- '{printf $1}'`
    local template_patient_id=`echo ${each_line}|awk -F, '{print $4}'`
    
    if [ "${leaveoneout}" = no ] || ( [ "${subject_image_id}" != "${template_left_image_id}" ] && [ "${subject_patient_id:0:10}" != "${template_patient_id:0:10}" ] )
    then 
      local left_corr=`${ffdevaluate} ${template_left_image} ${output_brain_image} -troi ${template_left_region} | grep Crosscorrelation| awk '{print $2}'`
      echo "left_corr=${left_corr}"
      echo "${left_corr},${template_left_image},${template_left_region}" >> ${output_left_corr}
    else
      echo "Skipping same subject ID ${template_left_image_id}..."
    fi 
  done
  
  echo rm -f ${output_brain_image} ${output_brain_image%.img}.hdr
  
  echo "brain-match done"
}

#
# Match the hippo and generate the cross-correlation with all the images in the library. 
#
function brain-shape-match()
{
  echo "number of arguments=$#"
  if [ $# != 10 ]
  then
    echo "brain shape match requires 10 arguments"
    exit
  fi

  local watjo_image=$1
  local watjo_brain_region=$2
  local subject_image=$3
  local subject_brain_region=$4
  local output_brain_image=$5
  local output_brain_region=$6
  local output_brain_series_number=$7
  local output_brain_air=$8
  local left_hippo_template_library=${9}
  local output_left_corr=${10}
  
  local subject_image_basename=`basename ${subject_image}`
  local subject_image_id=`echo ${subject_image_basename} | awk -F- '{printf $1}'`
  local subject_patient_id=`imginfo ${subject_image} -info | awk '{printf $5}'`
  
  if [ 1 = 1 ] 
  then
    if [ ! -f ${output_brain_air} ] 
    then 
    brain_to_brain_registration_without_repeat_mask ${watjo_image} ${watjo_brain_region} ${subject_image}  \
      ${output_brain_image} ${output_brain_series_number} ${output_brain_air} "no"
    fi 
    regslice ${output_brain_air} ${subject_brain_region} ${output_brain_region} 400 -i 2
  fi                      
  
  rm -f ${output_left_corr}
  local tmp_dir=`mktemp -d -q ~/temp/__maps-brain-shape-match-reg.XXXXXX`
  exec <  ${left_hippo_template_library} 
  while read each_line 
  do
    local template_left_image=${hippo_template_library_dir}/`echo ${each_line}|awk -F, '{print $1}'`
    local template_left_undilated_region=${hippo_template_library_dir}/`echo ${each_line}|awk -F, '{print $3}'`
    local template_left_image_basename=`basename ${template_left_image}`
    local template_left_image_id=`echo ${template_left_image_basename} | awk -F- '{printf $1}'`
    local template_patient_id=`imginfo ${template_left_image} -info | awk '{printf $5}'`
    
    if [ "${subject_image_id}" != "${template_left_image_id}" ] && [ "${subject_patient_id:0:10}" != "${template_patient_id:0:10}" ]
    then 
      makemask ${template_left_image} ${template_left_undilated_region} ${tmp_dir}/template_brain.img
      makemask ${template_left_image} ${output_brain_region} ${tmp_dir}/seg_brain.img
      local left_corr=`${seg_stat} -si ${tmp_dir}/template_brain.img ${tmp_dir}/seg_brain.img |tail -n 1 | awk '{printf $11}'`
      echo "left_corr=${left_corr}"
      echo "${left_corr},${template_left_image},${template_left_undilated_region}" >> ${output_left_corr}
    else
      echo "Skipping same subject ID ${template_left_image_id} or ${template_patient_id}..."
    fi 
  done
  rm -rf ${tmp_dir}
  
  echo rm -f ${output_brain_image} ${output_brain_image%.img}.hdr
  
  echo "brain-shape-match done"
}


#
# Delineate one hippo using FFD. 
#
function brain_delineation()
{
  echo "number of arguments=$#"
  if [ $# != 26 ]
  then
    echo "brain delineation requires 26 arguments"
    exit
  fi
  
  local subject_image=$1 
  local template_image=$2
  local template_brain_region=$3
  local output_areg_template_brain_image=$4
  local output_areg_template_brain_region=$5
  local output_areg_template_brain_series_number=$6
  local output_areg_template_air=$7
  local output_areg_hippo_region=${8}
  local output_delineate_dir=${9}
  local output_study_id=${10}
  local output_series_number=${11}
  local output_echo_number=${12}
  local output_local_areg_hippo_region=${13}
  local output_nreg_hippo_region=${14}
  local output_nreg_template_hippo_image=${15}
  local output_nreg_template_hippo_dof=${16}
  local output_areg_mm_hippo_region=${17}
  local output_nreg_mm_hippo_region=${18}
  local output_nreg_thresholded_hippo_region=${19}
  local nreg=${20}
  local f3d_brain_prereg=${21}
  local areg=${22}
  local control_point_spacing=${23}
  local f3d_iterations=${24}
  local template_vents_region=${25}
  local output_brain_with_vents=${26}

  if [ ! -f ${output_areg_hippo_region} ]
  then 
    # Brain to brain 12 dof registration
    if [ "${areg}" == "air" ]
    then 
      brain_to_brain_registration_without_repeat_mask_using_air ${template_image} ${template_brain_region} ${subject_image}   \
          ${output_areg_template_brain_image} ${output_areg_template_brain_series_number} ${output_areg_template_air} "yes"
      regslice ${output_areg_template_air} ${template_brain_region} ${output_areg_hippo_region} 500
    else 
      brain_to_brain_registration_without_repeat_mask_using_irtk ${template_image} ${template_brain_region} ${subject_image}   \
          ${output_areg_template_brain_image} ${output_areg_template_brain_series_number} ${output_areg_template_air}.dof "yes" "${init_9dof}"
	local tmp_dir=`mktemp -d -q ~/temp/__maps-f3d-reg.XXXXXX`
      ${ffdroitransformation} ${template_brain_region} ${output_areg_hippo_region} ${template_image}  ${subject_image} ${output_areg_template_air}.dof -invert -bspline -tmpdir ${tmp_dir}
      rm -rf ${tmp_dir}
    fi 
  fi
  
  if [ ! -f ${output_nreg_hippo_region} ]
  then 
    # F3D non-rigid registration. 
    local output_local_areg_template_brain_image=${output_areg_template_brain_image}
    local tmp_dir=`mktemp -d -q ~/temp/__maps-f3d-reg.XXXXXX`
  
    local output_local_areg_hippo_region_dilated_mask=${tmp_dir}/hippo-areg-mask-dilated.img
    makemask ${subject_image} ${output_local_areg_hippo_region} ${output_local_areg_hippo_region_dilated_mask} -d ${nreg_hippo_dilation}
    
    if [ "${nreg}" == "f3d" ] 
    then 
      local f3d_brain_prereg_flag=""
      if [ ! -f ${output_nreg_template_hippo_dof}.nii ]
      then 
        if [ "${f3d_brain_prereg}" == "yes" ]
        then 
          double_control_point_spacing=`echo "${control_point_spacing}*2" | bc`
          ${nifti_reg} -target ${subject_image} -source ${output_local_areg_template_brain_image} -cpp ${output_nreg_template_hippo_dof}.nii -result ${output_nreg_template_hippo_image} -ln 3 -lp 2 -sx ${double_control_point_spacing} -be ${f3d_bending_energy} -bin 128 -maxit ${f3d_iterations}
          ${nifti_resample} -target ${subject_image} -source ${output_local_areg_hippo_region_dilated_mask} -result ${output_local_areg_hippo_region_dilated_mask} -cpp ${output_nreg_template_hippo_dof}.nii -TRI
          local f3d_brain_prereg_flag="-incpp ${output_nreg_template_hippo_dof}.nii"
        fi   
        ${nifti_reg} -target ${subject_image} -source ${output_local_areg_template_brain_image} -tmask  ${output_local_areg_hippo_region_dilated_mask} -cpp ${output_nreg_template_hippo_dof}.nii -result ${output_nreg_template_hippo_image} -sx ${control_point_spacing} -ln 2 -be ${f3d_bending_energy} -nopy ${f3d_brain_prereg_flag} -bin 128 -maxit ${f3d_iterations}
      fi 
      
      ${abs_filter} -i ${output_nreg_template_hippo_image} -o ${output_nreg_template_hippo_image}
      local output_local_areg_hippo_region_mask=${tmp_dir}/hippo-areg-mask.img
      local output_nreg_hippo_mask=${tmp_dir}/hippo-nreg-mask.img
      makemask ${subject_image} ${output_local_areg_hippo_region} ${output_local_areg_hippo_region_mask} 
      ${nifti_resample} -target ${subject_image} -source ${output_local_areg_hippo_region_mask} -result ${output_nreg_hippo_mask} -cpp ${output_nreg_template_hippo_dof}.nii -TRI
      ${abs_filter} -i ${output_nreg_hippo_mask} -o ${output_nreg_hippo_mask}
      ${convert} -i ${output_nreg_hippo_mask} -o ${output_nreg_hippo_mask} -ot short
      makeroi -img ${output_nreg_hippo_mask} -out ${output_nreg_hippo_region} -alt 126
    else
      parameter_file=`mktemp ~/temp/param.XXXXXXXXXX`
      echo "# target image paramters      "  > ${parameter_file}
      echo "Target blurring (in mm) = 0   " >> ${parameter_file}
      echo "Target resolution (in mm) = 0 " >> ${parameter_file}
      echo "# source image paramters      " >> ${parameter_file}
      echo "Source blurring (in mm) = 0   " >> ${parameter_file}
      echo "Source resolution (in mm)  = 0" >> ${parameter_file}
      echo "# registration parameters     " >> ${parameter_file}
      echo "No. of resolution levels = 1  " >> ${parameter_file}
      echo "No. of bins = 128             " >> ${parameter_file}
      echo "No. of iterations = 20        " >> ${parameter_file}
      echo "No. of steps = 3              " >> ${parameter_file}
      echo "Length of steps = 12          " >> ${parameter_file}
      echo "Similarity measure = CC       " >> ${parameter_file}
      echo "Lambda = 0                    " >> ${parameter_file}
      echo "Interpolation mode = BSpline  " >> ${parameter_file}
      echo "Epsilon = 0.000001            " >> ${parameter_file}
      
      cat ${parameter_file}
      ${ffdnreg} ${subject_image} ${output_local_areg_template_brain_image} ${output_nreg_template_hippo_dof} -troi ${output_local_areg_hippo_region} \
        -dil 16 -gradient -inc 16 0 -nparams ${parameter_file}
      rm -f ${parameter_file}
      ${ffdtransformation} ${output_local_areg_template_brain_image} ${subject_image} ${output_nreg_template_hippo_image} ${output_nreg_template_hippo_dof} -bspline
      ${ffdroitransformation} ${output_local_areg_hippo_region} ${output_nreg_hippo_region}  ${output_areg_template_brain_image} ${subject_image} ${output_nreg_template_hippo_dof} -bspline
      ${ffdsubdivide} ${output_nreg_template_hippo_dof} ${output_nreg_template_hippo_dof}
    fi 
    
    rm -rf ${tmp_dir}
  fi 
  
  # Calculate mean brain intensity
  local mean_intensity=`imginfo ${subject_image} -av -roi ${output_nreg_hippo_region}`
  local threshold_70=`echo "${mean_intensity}*0.60" | bc`
  local threshold_160=`echo "${mean_intensity}*1.60" | bc`
  echo "Manual threshold=${threshold_70},${threshold_160}"
  
  temp_dir=`mktemp -d ~/temp/_hippo_mm.XXXXXX`
  local output_left_hippo_local_region_threshold_img=${temp_dir}/threshold.img
  local output_left_hippo_local_region_threshold=${temp_dir}/threshold
  local output_left_hippo_local_region_threshold_cd_img=${temp_dir}/threshold-cd.img
  local output_left_hippo_local_region_threshold_cd=${temp_dir}/threshold-cd
  
  if [ "${use_kmeans}" == "no" ]
  then 
    # Threshold. 
    makemask ${subject_image} ${output_nreg_hippo_region} ${output_left_hippo_local_region_threshold_img} -k -bpp 16
    makeroi -img ${output_left_hippo_local_region_threshold_img} -out ${output_left_hippo_local_region_threshold} \
      -alt ${threshold_70} -aut ${threshold_160}
    # erode once, followed by conditional dilation once between 60% and 160% of mean brain intensity. 
    #makemask ${subject_image} ${output_left_hippo_local_region_threshold} ${output_left_hippo_local_region_threshold_img} -e 1 
    #makeroi -img ${output_left_hippo_local_region_threshold_img} -out ${output_left_hippo_local_region_threshold} -alt 128
    makemask ${subject_image} ${output_left_hippo_local_region_threshold} ${output_left_hippo_local_region_threshold_img} -cd 2 60 160
    makeroi -img ${output_left_hippo_local_region_threshold_img} -out ${output_left_hippo_local_region_threshold} -alt 128
  else
    local output_left_hippo_local_region_img=${temp_dir}/region.img
    local init_1=`echo "${mean_intensity}*0.4" | bc -l`
    local init_2=`echo "${mean_intensity}*0.9" | bc -l`
    local init_3=`echo "${mean_intensity}*1.4" | bc -l`
    local threshold_160=`echo "${mean_intensity}*1.60" | bc -l`
    
    makemask ${subject_image} ${output_nreg_hippo_region} ${output_left_hippo_local_region_img} -d 3
    kmeans_output=`itkKmeansClassifierTest ${subject_image} ${output_left_hippo_local_region_img} ${temp_dir}/label1.img.gz ${temp_dir}/label2.img.gz 3 ${init_1} ${init_2} ${init_3}`
    
    echo "kmeans=${kmeans_output}"
    csf=`echo ${kmeans_output} | awk '{printf $1}'`
    csf_sd=`echo ${kmeans_output} | awk '{printf $2}'`
    gm=`echo ${kmeans_output} | awk '{printf $3}'`
    gm_sd=`echo ${kmeans_output} | awk '{printf $4}'`
    wm=`echo ${kmeans_output} | awk '{printf $5}'`
    wm_sd=`echo ${kmeans_output} | awk '{printf $6}'`
    
# Comments.     
: <<'COMMENTS'
    init_1=`echo "${mean_intensity}*0.4" | bc -l`
    init_2=`echo "${mean_intensity}*0.7" | bc -l`
    init_3=`echo "${mean_intensity}*0.9" | bc -l`
    init_4=`echo "${mean_intensity}*1.1" | bc -l`
    init_5=`echo "${mean_intensity}*1.4" | bc -l`
    
    kmeans_output=`itkKmeansClassifierTest ${subject_image} ${output_left_hippo_local_region_img} ${temp_dir}/label1.img.gz ${temp_dir}/label2.img.gz 5 ${init_1} ${init_2} ${init_3} ${init_4} ${init_5}`
    echo "kmeans=${kmeans_output}"
    csf=`echo ${kmeans_output} | awk '{printf $1}'`
    csf_sd=`echo ${kmeans_output} | awk '{printf $2}'`
    gm=`echo ${kmeans_output} | awk '{printf $5}'`
    gm_sd=`echo ${kmeans_output} | awk '{printf $6}'`
    wm=`echo ${kmeans_output} | awk '{printf $9}'`
    wm_sd=`echo ${kmeans_output} | awk '{printf $10}'`
    
    gm_csf=`echo ${kmeans_output} | awk '{printf $3}'`
COMMENTS    
    
# Comments.     
: <<'COMMENTS'
    if [ ! -f "${output_study_id}-tissues.nii" ]
    then 
      /var/NOT_BACKUP/work/NiftHippo-25/bin/bin/seg_EM -in ${subject_image} -mask ${output_left_hippo_local_region_img} -out ${output_study_id}-tissues.nii -nopriors 3 -mrf_beta 0 -v 1
    fi 
    /var/NOT_BACKUP/work/UCLToolkitDependencies/crkit-release-1.4-build/./tools/code/crlExtractSmallerImageFromImage -i ${output_study_id}-tissues.nii -a 3 -l 0 -m 1 -o ${output_study_id}-csf.nii
    niftkConvertImage -i ${output_study_id}-csf.nii -o ${output_study_id}-csf.img
    /var/NOT_BACKUP/work/UCLToolkitDependencies/crkit-release-1.4-build/./tools/code/crlExtractSmallerImageFromImage -i ${output_study_id}-tissues.nii -a 3 -l 1 -m 2 -o ${output_study_id}-gm.nii
    niftkConvertImage -i ${output_study_id}-gm.nii -o ${output_study_id}-gm.img
    /var/NOT_BACKUP/work/UCLToolkitDependencies/crkit-release-1.4-build/./tools/code/crlExtractSmallerImageFromImage -i ${output_study_id}-tissues.nii -a 3 -l 2 -m 3 -o ${output_study_id}-wm.nii
    niftkConvertImage -i ${output_study_id}-wm.nii -o ${output_study_id}-wm.img
    #niftkAdd -i gm.img -j wm.img -o gm_wm.img
    #makemask ${subject_image} ${output_nreg_hippo_region} mask.img -val 1
    #niftkMultiply -j mask.img -i gm_wm.img -o gm_wm_after_mask.img 
    #niftkThreshold -i gm_wm_after_mask.img  -o gm_wm_binary.img -l 0.5 -u 999 -in 255 -out 0 
    #makeroi -img gm_wm_binary.img -out ${output_left_hippo_local_region_threshold} -alt 128
    niftkThreshold -i ${output_study_id}-csf.img  -o ${output_study_id}-csf_binary.img -l 0.5 -u 999 -in 255 -out 0 
    makeroi -img ${output_study_id}-csf_binary.img -out ${output_study_id}-csf_binary -alt 128
    niftkThreshold -i ${output_study_id}-gm.img  -o ${output_study_id}-gm_binary.img -l 0.5 -u 999 -in 255 -out 0 
    makeroi -img ${output_study_id}-gm_binary.img -out ${output_study_id}-gm_binary -alt 128
    csf=`imginfo ${subject_image} -av -roi ${output_study_id}-csf_binary`
    gm=`imginfo ${subject_image} -av -roi ${output_study_id}-gm_binary`
    csf_sd=`imginfo ${subject_image} -sd -roi ${output_study_id}-csf_binary`
    gm_sd=`imginfo ${subject_image} -sd -roi ${output_study_id}-gm_binary`
    echo "csf=${csf}, gm=${gm}, csf_sd=${csf_sd}, gm_sd=${gm_sd}"
    # k-means is actually not bad, compared to EM.  
COMMENTS
    
    number_of_gm_sd_70=1.04  # CI=70%
    number_of_gm_sd_80=1.28   # CI=80%
    number_of_gm_sd_85=1.44   # CI=85%
    number_of_gm_sd_875=1.53   # CI=87.5%
    number_of_gm_sd_90=1.64  # CI=90%
    number_of_gm_sd_95=1.96  # CI=95%
    number_of_gm_160_sd=1.7
    
    distance_csf_gm=`echo "${gm}-${csf}" | bc -l`
    distance_factor=`echo "${gm_sd}/(${gm_sd}+${csf_sd})" | bc -l`
    lower_threshold_distance=`echo "${gm}-${distance_factor}*${distance_csf_gm}" | bc -l`
    lower_threshold_sd=`echo "${gm}-${number_of_gm_160_sd}*${gm_sd}" | bc -l`
    lower_threshold_95=${lower_threshold_distance}
    upper_threshold_95=`echo "${wm}+4.42*${wm_sd}" | bc -l`
    
    echo "lower_threshold_distance=${lower_threshold_distance}, lower_threshold_sd=${lower_threshold_sd}"
    
    makemask ${subject_image} ${output_nreg_hippo_region} ${output_left_hippo_local_region_threshold_img} -k -bpp 16
    makeroi -img ${output_left_hippo_local_region_threshold_img} -out ${output_left_hippo_local_region_threshold} \
            -alt ${lower_threshold_95} -aut ${upper_threshold_95}
    makemask ${subject_image} ${output_left_hippo_local_region_threshold} ${output_left_hippo_local_region_threshold_img} 
    niftkConnectedComponents ${output_left_hippo_local_region_threshold_img} ${output_left_hippo_local_region_threshold} img -largest
    makeroi -img ${output_left_hippo_local_region_threshold_img} -out ${output_left_hippo_local_region_threshold} -alt 0
      
    local new_mean_intensity=`imginfo ${subject_image} -av -roi ${output_left_hippo_local_region_threshold}`
    lower_threshold=${lower_threshold_95}
    upper_threshold=${upper_threshold_95}
    lower_threshold_percent=`echo "(100*${lower_threshold})/${new_mean_intensity}" | bc -l`
    upper_threshold_percent=`echo "(100*${upper_threshold})/${new_mean_intensity}" | bc -l`
    threshold_160_percent=`echo "(100*${threshold_160})/${new_mean_intensity}" | bc -l`
    threshold_70_percent=`echo "(100*${threshold_70})/${new_mean_intensity}" | bc -l`

    echo "percent=${lower_threshold_percent},${upper_threshold_percent},${threshold_70_percent},${threshold_160_percent}"
    
    if [ "${cd_mode}" == "2" ]
    then 
      echo "cd_mode=2"
      makemask ${subject_image} ${output_left_hippo_local_region_threshold} ${output_left_hippo_local_region_threshold_img} -cd 2 ${lower_threshold_percent} ${upper_threshold_percent}
      makeroi -img ${output_left_hippo_local_region_threshold_img} -out ${output_left_hippo_local_region_threshold} -alt 128
    else
      echo "cd_mode=${cd_mode}"
      makemask ${subject_image} ${output_left_hippo_local_region_threshold} ${output_left_hippo_local_region_threshold_img} -cd 1 ${lower_threshold_percent} ${upper_threshold_percent}
      makeroi -img ${output_left_hippo_local_region_threshold_img} -out ${output_left_hippo_local_region_threshold} -alt 128
      
      local new_mean_intensity=`imginfo ${subject_image} -av -roi ${output_left_hippo_local_region_threshold}`
      lower_threshold_percent=`echo "(100*${lower_threshold})/${new_mean_intensity}" | bc -l`
      upper_threshold_percent=`echo "(100*${upper_threshold})/${new_mean_intensity}" | bc -l`
      makemask ${subject_image} ${output_left_hippo_local_region_threshold} ${output_left_hippo_local_region_threshold_img} -cd 1 ${lower_threshold_percent} ${upper_threshold_percent}
      makeroi -img ${output_left_hippo_local_region_threshold_img} -out ${output_left_hippo_local_region_threshold} -alt 128
    fi 
    
  fi 
  
  
  # Add in the vents, if needed. 
  if [ -f "${template_vents_region}"  ]
  then 
      local tmp_vents_dir=`mktemp -d -q ~/temp/__maps-f3d-vents.XXXXXX`
      local output_areg_vents_region=${tmp_vents_dir}/areg-vents
      local output_local_areg_vents_region_mask=${tmp_vents_dir}/areg-vents.img
      local output_nreg_vents_mask=${tmp_vents_dir}/nreg-vents.img
      local output_nreg_vents_region=${tmp_vents_dir}/nreg-vents
      local temp_brain_with_vents_region=${temp_dir}/brain_and_vents
      local temp_brain_with_vents_region_img=${temp_dir}/brain_and_vents.img
      
      local tmp_dir=`mktemp -d -q ~/temp/__maps-f3d-reg.XXXXXX`
      ${ffdroitransformation} ${template_vents_region} ${output_areg_vents_region} ${template_image}  ${subject_image} ${output_areg_template_air}.dof -invert -bspline -tmpdir ${tmp_dir}
      rm -rf ${tmp_dir}
      
      makemask ${subject_image} ${output_areg_vents_region} ${output_local_areg_vents_region_mask} 
      ${nifti_resample} -target ${subject_image} -source ${output_local_areg_vents_region_mask} -result ${output_nreg_vents_mask} -cpp ${output_nreg_template_hippo_dof}.nii -TRI
      ${abs_filter} -i ${output_nreg_vents_mask} -o ${output_nreg_vents_mask}
      ${convert} -i ${output_nreg_vents_mask} -o ${output_nreg_vents_mask} -ot short
      makeroi -img ${output_nreg_vents_mask} -out ${output_nreg_vents_region} -alt 126
      addRegions.sh ${subject_image} ${output_left_hippo_local_region_threshold} ${output_nreg_vents_region} ${temp_brain_with_vents_region} ""
      rm -rf ${tmp_vents_dir}
      
      # Only take the largest connected component. 
      makemask ${subject_image} ${temp_brain_with_vents_region} ${temp_brain_with_vents_region_img} 
      niftkConnectedComponents ${temp_brain_with_vents_region_img} ${temp_brain_with_vents_region} img -largest
      makeroi -img ${temp_brain_with_vents_region_img} -out ${temp_brain_with_vents_region} -alt 0
      cp ${temp_brain_with_vents_region} ${output_brain_with_vents}
  fi 
  
  # Only take the largest connected component. 
  makemask ${subject_image} ${output_left_hippo_local_region_threshold} ${output_left_hippo_local_region_threshold_img} 
  niftkConnectedComponents ${output_left_hippo_local_region_threshold_img} ${output_left_hippo_local_region_threshold} img -largest
  makeroi -img ${output_left_hippo_local_region_threshold_img} -out ${output_left_hippo_local_region_threshold} -alt 0
  
  cp ${output_left_hippo_local_region_threshold} ${output_nreg_thresholded_hippo_region}

  rm -rf ${temp_dir}
  
  return 0
}


#
# Reorient the region back to the original image. 
#
function reorient_region()
{
  local subject_image=${1}
  local region=${2}
  local original_subject_image=${3}
  local oriented_region=${4}
  
  local dims=`imginfo ${subject_image} -dims | awk '{printf "%d %d %d", $1, $2, $3}'`
  local orient=`imginfo ${original_subject_image} -orient`
  
  if [ "${orient}" != "sag" ]
  then 
    regchange ${region} ${oriented_region} ${dims} -orient sag ${orient}
  fi 
}

#
# Delineate one hippo by stapling multiple segmentations. 
#
function brain-delineation-using-staple()
{
  local left_or_right=${1}
  local output_left_corr=${2}
  local subject_image=${3}
  local hippo_template_library_dir=${4}
  local hippo_template_library_original=${5}
  local output_left_dir=${6}
  local mrf_weighting=${7}
  local nreg=${8}
  local f3d_brain_prereg=${9}
  local areg=${10}
  local control_point_spacing=${11}
  local f3d_iterations=${12}
  local vents_or_not=${13}
  local remove_dir=${14}
  local use_orientation=${15}
  local original_subject_image=${16}
  
  local subject_image_basename=`basename ${subject_image}`
  local subject_image_id=`echo ${subject_image_basename} | awk -F- '{printf $1}'`
  local subject_brain_region=${subject_image_id}-brain-region
  local template_vents_region="__dummy__"
  
  # Select the best match ones from the library - including flipped images. 
  local temp_left_corr_file=`mktemp ~/temp/left_corr.XXXXXX`
  cat ${output_left_corr} | sort -nr | head -n ${staple_count}  > ${temp_left_corr_file}
  cat ${temp_left_corr_file}
  
  local staple_command_line_nreg_thresholded=""
  local staple_command_line_brain_with_vents=""
  local results_dir=""
  exec < ${temp_left_corr_file} 
  while read each_line
  do 
    # A bunch of messy code to find out the template image. 
    echo ${each_line}
    local left_image=`echo ${each_line} | awk -F, '{printf $2}'`
    local left_image_basename=`basename ${left_image}`
    echo ${left_image_basename}
    
    local template_left_image=""
    exec 6<&0
    exec < ${hippo_template_library_original}
    while read line 
    do
      local each_image=`echo ${line}|awk -F, '{print $2}'`
      
      if [ ${each_image} = ${left_image_basename} ] 
      then 
        local template_left_image=${hippo_template_library_dir}/`echo ${line}|awk -F, '{print $8}'`
        local template_left_brain_region=${hippo_template_library_dir}/`echo ${line}|awk -F, '{print $9}'`
        if [ "${vents_or_not}" == "yes" ]
        then 
          local template_vents_region=${hippo_template_library_dir}/`echo ${line}|awk -F, '{print $12}'`
        fi 
      fi 
    done
    exec 0<&6 6<&-
    if [ ! -f "${template_left_image}" ]
    then
      echo "Cannot locate original image for ${left_image}"
      exit
    fi
    
    echo ${template_left_image},${template_left_brain_region}
    local template_prefix=`basename ${template_left_image} | awk -F. '{print $1}'`
    
    # Create the directory structure
    local output_left_delineate_dir=${output_left_dir}/delineate/${output_study_id}-${template_prefix}
    
    mkdir -p ${output_left_delineate_dir} 
    
    local output_areg_template_brain_image=${output_left_delineate_dir}/${output_study_id}-${output_areg_template_brain_series_number}-${output_left_echo_number}.img
    local output_areg_template_brain_region=${output_left_delineate_dir}/`basename ${subject_brain_region}`
    local output_areg_template_air=${output_left_delineate_dir}/${output_study_id}.air
    local output_left_image=${output_left_delineate_dir}/${output_study_id}-${output_left_series_number}-${output_left_echo_number}.img
    local output_areg_hippo_region=${output_left_delineate_dir}/`basename ${subject_brain_region}`-areg
    local output_local_areg_hippo_region=${output_left_delineate_dir}/`basename ${subject_brain_region}`-areg
    local output_nreg_hippo_region=${output_left_delineate_dir}/`basename ${subject_brain_region}`-nreg
    local output_nreg_template_hippo_image=${output_left_delineate_dir}/${output_study_id}-${output_left_series_number}-${output_left_echo_number}-hippo.img
    local output_nreg_template_hippo_dof=${output_left_delineate_dir}/${output_study_id}-${output_left_series_number}-${output_left_echo_number}-hippo.dof
    local output_areg_mm_hippo_region=${output_left_delineate_dir}/`basename ${subject_brain_region}`-areg-mm
    local output_nreg_mm_hippo_region=${output_left_delineate_dir}/`basename ${subject_brain_region}`-nreg-mm
    local output_nreg_thresholded_hippo_region=${output_left_delineate_dir}/`basename ${subject_brain_region}`-nreg-thresholded
    local output_brain_with_vents=${output_left_delineate_dir}/`basename ${subject_brain_region}`-brain-and-vents
  
    # Delineate the hippo.   
    brain_delineation ${subject_image} ${template_left_image} ${template_left_brain_region} \
                      ${output_areg_template_brain_image} ${output_areg_template_brain_region} \
                      ${output_areg_template_brain_series_number} ${output_areg_template_air} \
                      ${output_areg_hippo_region} \
                      ${output_left_delineate_dir} ${output_study_id} ${output_left_series_number} ${output_left_echo_number}  \
                      ${output_local_areg_hippo_region} ${output_nreg_hippo_region} ${output_nreg_template_hippo_image} \
                      ${output_nreg_template_hippo_dof} ${output_areg_mm_hippo_region} ${output_nreg_mm_hippo_region} ${output_nreg_thresholded_hippo_region} \
                      ${nreg} ${f3d_brain_prereg} ${areg} ${control_point_spacing} ${f3d_iterations} ${template_vents_region} ${output_brain_with_vents}
                      
    rm -f ${output_areg_template_brain_image} ${output_nreg_template_hippo_image}                      
                      
    makemask ${subject_image} ${output_nreg_thresholded_hippo_region} ${output_nreg_thresholded_hippo_region}.img -val 1
    local staple_command_line_nreg_thresholded="${staple_command_line_nreg_thresholded} ${output_nreg_thresholded_hippo_region}.img"
    if [ "${vents_or_not}" == yes ]
    then 
      makemask ${subject_image} ${output_brain_with_vents} ${output_brain_with_vents}.img -val 1
      local staple_command_line_brain_with_vents="${staple_command_line_brain_with_vents} ${output_brain_with_vents}.img"
    fi 
    local results_dir="${results_dir} ${output_left_delineate_dir}"
  done
  rm -f ${temp_left_corr_file}
  
  local output_hippo_areg_region=${output_left_dir}/delineate/${output_study_id}-${output_areg_template_brain_series_number}-${output_left_echo_number}-hippo-areg
  local output_hippo_staple_nreg_thresholded_region=${output_left_dir}/delineate/${output_study_id}-${output_areg_template_brain_series_number}-${output_left_echo_number}-hippo-staple-nreg-thresholded
  local output_hippo_staple_nreg_thresholded_staple=${output_hippo_staple_nreg_thresholded_region}-staple.img
  local output_hippo_staple_nreg_thresholded=${output_hippo_staple_nreg_thresholded_region}.img
  
  local output_hippo_staple_weights=${output_left_dir}/delineate/${output_study_id}-weights.nrrd
  local output_hippo_staple_foregroundprob=${output_left_dir}/delineate/${output_study_id}-foreground-prob.img
  local output_hippo_staple_mf_weights=${output_left_dir}/delineate/${output_study_id}-mfweights.nrrd
  local output_hippo_staple_mf_region=${output_left_dir}/delineate/${output_study_id}-staple-mrf
  local output_hippo_staple_mf_seg=${output_hippo_staple_mf_region}.img
  local output_hippo_staple_weights_thresholded=${output_left_dir}/delineate/${output_study_id}-weights-thresholded.nrrd
  
  if [ 1 == 0 ] 
  then 
  ${crlSTAPLE} --outputImage ${output_hippo_staple_weights} ${staple_command_line_nreg_thresholded} 
  ${crlExtractSmallerImageFromImage} -i ${output_hippo_staple_weights} -o ${output_hippo_staple_nreg_thresholded} -l 1 -m 2 -a 3
  ${threshold} -i ${output_hippo_staple_nreg_thresholded} -o ${output_hippo_staple_nreg_thresholded} -u 2 -l ${staple_confidence} -in 255 -out 0
  ${convert} -i ${output_hippo_staple_nreg_thresholded} -o ${output_hippo_staple_nreg_thresholded} -ot short
  niftkConnectedComponents ${output_hippo_staple_nreg_thresholded} ${output_hippo_staple_nreg_thresholded%.img} img -largest
  makeroi -img ${output_hippo_staple_nreg_thresholded} -out ${output_hippo_staple_nreg_thresholded_region}-${staple_confidence}-${count} -alt 0
  
  # reset the probability to 0.5 if it is less than the staple_confidence, in order to allow the MRF not too smooth the segmentation. 
  ${thresholdProb} ${output_hippo_staple_weights} ${staple_confidence} 0.5 ${output_hippo_staple_weights_thresholded}
  
  ${crlMeanFieldMRF} ${output_hippo_staple_weights_thresholded} automatic 0.00001 ${mrf_weighting} 5 ${output_hippo_staple_mf_weights}
  ${crlExtractSmallerImageFromImage} -i ${output_hippo_staple_mf_weights} -o ${output_hippo_staple_mf_seg} -l 1 -m 2 -a 3
  ${threshold} -i ${output_hippo_staple_mf_seg} -o ${output_hippo_staple_mf_seg} -u 2 -l ${staple_confidence} -in 255 -out 0
  ${convert} -i ${output_hippo_staple_mf_seg} -o ${output_hippo_staple_mf_seg} -ot short
  niftkConnectedComponents ${output_hippo_staple_mf_seg} ${output_hippo_staple_mf_seg%.img} img -largest
  makeroi -img ${output_hippo_staple_mf_seg} -out ${output_hippo_staple_mf_region} -alt 0
  
  cp ${output_hippo_staple_mf_region} ${output_hippo_staple_mf_region}_${staple_count}_${mrf_weighting}
    
  local dims=`imginfo ${subject_image} -dims | awk '{printf $1" "$2" "$3}'`
  
  local image_info=`imginfo ${subject_image} -info`
  local name=`echo ${image_info}|awk '{print $5}'`
  local study_id=`echo ${image_info}|awk '{print $1}'`
  local series_no=`echo ${image_info}|awk '{print $2}'`
  local scan_date=`imginfo ${subject_image} -info -datefmt|awk '{print $4 " " $5 " " $6 " " $7}'`
  local output_region_name=. 
  local quality=MAPS
  local structure="Brain"
  local segmentor=MAPS
  
  regchange ${output_hippo_staple_mf_region} ${output_left_dir}/delineate/. ${dims} -study ${study_id} -series ${series_no} -name "${name}" -segmentor ${segmentor} -structure ${structure} -quality ${quality} -acq "${scan_date}" -c
  fi 
  
  # VOTE
  local output_hippo_staple_nreg_thresholded_vote_img=${output_hippo_staple_nreg_thresholded_region}-vote-${staple_count}.img
  local output_hippo_staple_nreg_thresholded_vote_region=${output_hippo_staple_nreg_thresholded_region}-vote-${staple_count}
  ${combine} VOTE 0 1 ${output_hippo_staple_nreg_thresholded_vote_img} ${staple_command_line_nreg_thresholded}
  niftkConnectedComponents ${output_hippo_staple_nreg_thresholded_vote_img} ${output_hippo_staple_nreg_thresholded_vote_img%.img} img -largest
  makeroi -img ${output_hippo_staple_nreg_thresholded_vote_img} -out ${output_hippo_staple_nreg_thresholded_vote_region} -alt 0
  rm -f ${output_hippo_staple_nreg_thresholded_vote_img} ${output_hippo_staple_nreg_thresholded_vote_img%.img}.hdr
  if [ "${use_orientation}" = "yes" ] 
  then 
    reorient_region ${subject_image} ${output_hippo_staple_nreg_thresholded_vote_region} ${original_subject_image} ${output_hippo_staple_nreg_thresholded_vote_region}
  fi 

  if [ 1 == 0 ]
  then 
    # SBA - median
    local output_hippo_staple_nreg_thresholded_sba_img=${output_hippo_staple_nreg_thresholded_region}-sba-median-${staple_count}.img
    local output_hippo_staple_nreg_thresholded_sba_region=${output_hippo_staple_nreg_thresholded_region}-sba-median-${staple_count}
    ${combine} SBA 0 1 ${output_hippo_staple_nreg_thresholded_sba_img} ${staple_command_line_nreg_thresholded}
    niftkConnectedComponents ${output_hippo_staple_nreg_thresholded_sba_img} ${output_hippo_staple_nreg_thresholded_sba_img%.img} img -largest
    makeroi -img ${output_hippo_staple_nreg_thresholded_sba_img} -out ${output_hippo_staple_nreg_thresholded_sba_region} -alt 0
    rm -f ${output_hippo_staple_nreg_thresholded_sba_img} ${output_hippo_staple_nreg_thresholded_sba_img%.img}.hdr
  fi 
  
  # SBA - mean
  local output_hippo_staple_nreg_thresholded_sba_img=${output_hippo_staple_nreg_thresholded_region}-sba-mean-${staple_count}.img
  local output_hippo_staple_nreg_thresholded_sba_region=${output_hippo_staple_nreg_thresholded_region}-sba-mean-${staple_count}
  local output_hippo_staple_nreg_thresholded_sba_region_dilated=${output_hippo_staple_nreg_thresholded_region}-sba-mean-${staple_count}-dilated  
  ${combine} SBA 0 2 ${output_hippo_staple_nreg_thresholded_sba_img} ${staple_command_line_nreg_thresholded}
  niftkConnectedComponents ${output_hippo_staple_nreg_thresholded_sba_img} ${output_hippo_staple_nreg_thresholded_sba_img%.img} img -largest
  makeroi -img ${output_hippo_staple_nreg_thresholded_sba_img} -out ${output_hippo_staple_nreg_thresholded_sba_region} -alt 0
  rm -f ${output_hippo_staple_nreg_thresholded_sba_img} ${output_hippo_staple_nreg_thresholded_sba_img%.img}.hdr
  
  
###
### Testing extra erosion and conditional dilation. 
###
  if [ "${use_kmeans}" == "yes" ]
  then 
  
    temp_dir=`mktemp -d ~/temp/_hippo_mm.XXXXXX`
    local output_left_hippo_local_region_threshold_img=${temp_dir}/threshold.img
    local output_left_hippo_local_region_threshold=${temp_dir}/threshold
    local output_left_hippo_local_region_threshold_cd_img=${temp_dir}/threshold-cd.img
    local output_left_hippo_local_region_threshold_cd=${temp_dir}/threshold-cd
    local output_left_hippo_local_region_img=${temp_dir}/region.img
    
    mean_intensity=`imginfo ${subject_image} -av -roi ${output_hippo_staple_nreg_thresholded_sba_region}`
    threshold_70=`echo "${mean_intensity}*0.60" | bc`
    threshold_160=`echo "${mean_intensity}*1.60" | bc`
    echo "Manual threshold=${threshold_70},${threshold_160}"
    
    
    local init_1=`echo "${mean_intensity}*0.4" | bc -l`
    local init_2=`echo "${mean_intensity}*0.9" | bc -l`
    local init_3=`echo "${mean_intensity}*1.4" | bc -l`
    local threshold_160=`echo "${mean_intensity}*1.60" | bc -l`
      
    makemask ${subject_image} ${output_hippo_staple_nreg_thresholded_sba_region} ${output_left_hippo_local_region_img} -d 3
    kmeans_output=`itkKmeansClassifierTest ${subject_image} ${output_left_hippo_local_region_img} ${temp_dir}/label1.img.gz ${temp_dir}/label2.img.gz 3 ${init_1} ${init_2} ${init_3}`
      
    echo "kmeans=${kmeans_output}"
    csf=`echo ${kmeans_output} | awk '{printf $1}'`
    csf_sd=`echo ${kmeans_output} | awk '{printf $2}'`
    gm=`echo ${kmeans_output} | awk '{printf $3}'`
    gm_sd=`echo ${kmeans_output} | awk '{printf $4}'`
    wm=`echo ${kmeans_output} | awk '{printf $5}'`
    wm_sd=`echo ${kmeans_output} | awk '{printf $6}'`

    number_of_gm_sd_70=1.04  # CI=70%
    number_of_gm_sd_80=1.28   # CI=80%
    number_of_gm_sd_85=1.44   # CI=85%
    number_of_gm_sd_875=1.53   # CI=87.5%
    number_of_gm_sd_90=1.64  # CI=90%
    number_of_gm_sd_95=1.96  # CI=95%
    number_of_gm_160_sd=1.7
    
    distance_csf_gm=`echo "${gm}-${csf}" | bc -l`
    distance_factor=`echo "${gm_sd}/(${gm_sd}+${csf_sd})" | bc -l`
    lower_threshold_distance=`echo "${gm}-${distance_factor}*${distance_csf_gm}" | bc -l`
    lower_threshold_sd=`echo "${gm}-${number_of_gm_160_sd}*${gm_sd}" | bc -l`
    lower_threshold_95=${lower_threshold_distance}
    upper_threshold_95=`echo "${wm}+4.42*${wm_sd}" | bc -l`
    echo "lower_threshold_distance=${lower_threshold_distance}, lower_threshold_sd=${lower_threshold_sd}, lower_threshold_95=${lower_threshold_95}"
    
    makemask ${subject_image} ${output_hippo_staple_nreg_thresholded_sba_region} ${output_left_hippo_local_region_threshold_img} -k -bpp 16
    makeroi -img ${output_left_hippo_local_region_threshold_img} -out ${output_left_hippo_local_region_threshold} \
            -alt ${lower_threshold_95} -aut ${upper_threshold_95}
    makemask ${subject_image} ${output_left_hippo_local_region_threshold} ${output_left_hippo_local_region_threshold_img} -e 1 
    niftkConnectedComponents ${output_left_hippo_local_region_threshold_img} ${output_left_hippo_local_region_threshold} img -largest
    makeroi -img ${output_left_hippo_local_region_threshold_img} -out ${output_left_hippo_local_region_threshold} -alt 0
      
    local new_mean_intensity=`imginfo ${subject_image} -av -roi ${output_left_hippo_local_region_threshold}`
    lower_threshold=${lower_threshold_95}
    upper_threshold=${upper_threshold_95}
    lower_threshold_percent=`echo "(100*${lower_threshold})/${new_mean_intensity}" | bc -l`
    upper_threshold_percent=`echo "(100*${upper_threshold})/${new_mean_intensity}" | bc -l`
    threshold_160_percent=`echo "(100*${threshold_160})/${new_mean_intensity}" | bc -l`
    threshold_70_percent=`echo "(100*${threshold_70})/${new_mean_intensity}" | bc -l`

    echo "percent=${lower_threshold_percent},${upper_threshold_percent},${threshold_70_percent},${threshold_160_percent}"
      
    makemask ${subject_image} ${output_left_hippo_local_region_threshold} ${output_left_hippo_local_region_threshold_img} -cd 2 ${lower_threshold_percent} ${upper_threshold_percent}
    makeroi -img ${output_left_hippo_local_region_threshold_img} -out ${output_left_hippo_local_region_threshold} -alt 128
    
    cp ${output_left_hippo_local_region_threshold} ${output_hippo_staple_nreg_thresholded_sba_region}_mm
    rm -rf ${temp_dir}
  fi 
  
  
###
###
###  
  
  
  makemask ${subject_image} ${output_hippo_staple_nreg_thresholded_sba_region} ${output_hippo_staple_nreg_thresholded_sba_img} -d 2
  makeroi -img ${output_hippo_staple_nreg_thresholded_sba_img} -out ${output_hippo_staple_nreg_thresholded_sba_region_dilated} -alt 0
  rm -f ${output_hippo_staple_nreg_thresholded_sba_img} ${output_hippo_staple_nreg_thresholded_sba_img%.img}.hdr
  if [ "${use_orientation}" = "yes" ] 
  then 
    reorient_region ${subject_image} ${output_hippo_staple_nreg_thresholded_sba_region} ${original_subject_image} ${output_hippo_staple_nreg_thresholded_sba_region}
    reorient_region ${subject_image} ${output_hippo_staple_nreg_thresholded_sba_region_dilated} ${original_subject_image} ${output_hippo_staple_nreg_thresholded_sba_region_dilated}
  fi 
  
  if [ "${vents_or_not}" == "yes" ]
  then 
    local output_brain_with_vents_sba_img=${output_hippo_staple_nreg_thresholded_region}-brain-with-vents-sba-mean-${staple_count}.img
    local output_brain_with_vents_sba_region=${output_hippo_staple_nreg_thresholded_region}-brain-with-vents-sba-mean-${staple_count}
    ${combine} SBA 0 2 ${output_brain_with_vents_sba_img} ${staple_command_line_brain_with_vents}
    niftkConnectedComponents ${output_brain_with_vents_sba_img} ${output_brain_with_vents_sba_img%.img} img -largest
    makeroi -img ${output_brain_with_vents_sba_img} -out ${output_brain_with_vents_sba_region} -alt 0
    rm -f ${output_brain_with_vents_sba_img} ${output_brain_with_vents_sba_img%.img}.hdr
    if [ "${use_orientation}" = "yes" ] 
    then 
      reorient_region ${subject_image} ${output_brain_with_vents_sba_region} ${original_subject_image} ${output_brain_with_vents_sba_region}
    fi 
  fi 
    
  rm -f ${output_hippo_staple_nreg_thresholded_staple} ${output_hippo_staple_nreg_thresholded_staple%.img}.hdr ${output_hippo_staple_nreg_thresholded} ${output_hippo_staple_nreg_thresholded%.img}.hdr
  rm -f ${output_hippo_staple_weights} ${output_hippo_staple_mf_weights} ${output_hippo_staple_mf_seg} ${output_hippo_staple_mf_seg%.img}.hdr
  
  rm -f ${staple_command_line_nreg_thresholded}
  rm -f ${output_areg_mm_hippo_region}.img ${output_areg_mm_hippo_region}.hdr
  
  if [ "${remove_dir}" == "yes" ] && [ -f "${output_hippo_staple_nreg_thresholded_sba_region}" ] 
  then 
    if [ "${vents_or_not}" == "no" ] || [ -f "${output_brain_with_vents_sba_region}" ]
    then 
      rm -rf ${results_dir}
    fi 
  fi 
}




echo "Starting on `hostname` on `date`..."
echo "Arguments: $*"

hippo_template_library_dir=$1
subject_image=$2
output_study_id=$3
output_left_dir=$4
input_staple_count_start=$5
input_staple_count_end=$6
mrf_weighting=$7
left_hippo_template_library=${8}
hippo_template_library_original=${9}
watjo_image=${10}
watjo_brain_region=${11}
nreg_hippo_dilation=${12}
nreg=${13}
f3d_brain_prereg=${14}
areg=${15}
control_point_spacing=${16}
f3d_bending_energy=${17}
f3d_iterations=${18}
staple_confidence=${19}
vents_or_not=${20}
remove_dir=${21}
use_orientation=${22}
leaveoneout=${23}
use_kmeans=${24}
init_9dof=${25}
cd_mode=${26}

output_areg_template_brain_series_number=400
output_left_series_number=665
output_left_echo_number=1

original_subject_image="dummy"

# Create the directory structure
output_left_match_dir=${output_left_dir}/match
mkdir -p ${output_left_match_dir} 

if [ "${use_orientation}" == "yes" ] 
then 
  original_orietation=`imginfo ${subject_image} -orient`
  original_subject_image=${subject_image}
  if [ "${original_orietation}" != "sag" ]
  then 
    reoriented_subject_image=${output_left_match_dir}/${output_study_id}-002-1_sag.img
    anchange ${subject_image} ${reoriented_subject_image} -orient sag
    subject_image=${reoriented_subject_image}
  fi 
fi 

output_brain_image=${output_left_match_dir}/${output_study_id}-${output_areg_template_brain_series_number}-${output_left_echo_number}.img
output_brain_region=${output_left_match_dir}/${output_study_id}-output-brain-region
output_brain_air=${output_left_match_dir}/${output_study_id}.air

output_left_corr=${output_left_match_dir}/${output_study_id}-corr.txt

if [ ! -f ${output_left_corr} ]
then 
brain-match ${watjo_image} ${watjo_brain_region} ${subject_image} \
            ${output_brain_image} ${output_areg_template_brain_series_number} ${output_brain_air}  \
            ${left_hippo_template_library} ${output_left_corr} ${areg} ${leaveoneout}
            
fi             

for ((count=${input_staple_count_end}; count >= ${input_staple_count_start}; count-=2))
do 
  staple_count=${count}
  brain-delineation-using-staple left ${output_left_corr} ${subject_image} \
    ${hippo_template_library_dir} ${hippo_template_library_original} ${output_left_dir} ${mrf_weighting} \
    ${nreg} ${f3d_brain_prereg} ${areg} ${control_point_spacing} ${f3d_iterations} ${vents_or_not} \
    ${remove_dir} ${use_orientation} ${original_subject_image}
done

if [ "${use_orientation}" == "yes" ] 
then 
  if [ "${original_orietation}" != "sag" ]
  then 
    rm -f ${reoriented_subject_image} ${reoriented_subject_image%.img}.hdr
  fi 
fi 
        
output_left_shape_corr=${output_left_match_dir}/${output_study_id}-shape-corr.txt
output_hippo_staple_mf_region=${output_left_dir}/delineate/${output_study_id}-staple-mrf
output_brain_region=${output_left_dir}/delineate/${output_study_id}-shape-match-brain
        
# Re-select the atlases again using the current segmentation.         
echo brain-shape-match ${watjo_image} ${watjo_brain_region} ${subject_image} ${output_hippo_staple_mf_region} \
            ${output_brain_image} ${output_brain_region} ${output_areg_template_brain_series_number} ${output_brain_air}  \
            ${left_hippo_template_library} ${output_left_shape_corr} 
            
echo brain-delineation-using-staple left ${output_left_shape_corr} ${subject_image} \
        ${hippo_template_library_dir} ${hippo_template_library_original} ${output_left_dir} ${mrf_weighting} ${nreg}
            
echo rm -f ${output_brain_image} ${output_brain_image%.img}.hdr


echo "Finished on `hostname` on `date`..."



