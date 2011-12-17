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
#  Last Changed      : $LastChangedDate: 2010-05-28 22:05:02 +0100 (Fri, 28 May 2010) $ 
#  Revision          : $Revision: 3326 $
#  Last modified by  : $Author: mjc $
#
#  Original author   : leung@drc.ion.ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/

# Source the common function. 
source MAPS-common.sh

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
export ffdnreg=${MIDAS_FFD}/ffdnreg.sh
export ffdtransformation=${MIDAS_FFD}/ffdtransformation.sh
export ffdroitransformation=${MIDAS_FFD}/ffdroitransformation.sh
export ffdsubdivide=${FFITK_BIN}/ffdsubdivide
export threshold=niftkThreshold
export nifti_reg=${F3D_BIN}/reg_f3d
export nifti_resample=${F3D_BIN}/reg_resample


atlas_image=$1
atlas_brain_region=$2
atlas_left_hippo_region=$3
atlas_right_hippo_region=$4
subject_image=$5
subject_brain_region=$6
output_study_id=$7
output_left_dir=$8
output_right_dir=$9

output_echo_number=1
output_areg_template_brain_series_number=400

if [ "${atlas_left_hippo_region}" != "dummy" ] 
then 
  output_delineate_dir=${output_left_dir}
  output_series_number=555
  atlas_hippo_region=${atlas_left_hippo_region}
  
  output_areg_template_brain_image=${output_delineate_dir}/${output_study_id}-${output_areg_template_brain_series_number}-${output_echo_number}.img
  output_areg_template_brain_region=${output_delineate_dir}/`basename ${subject_brain_region}`
  output_areg_template_air=${output_delineate_dir}/${output_study_id}.air
  output_areg_hippo_region=${output_delineate_dir}/`basename ${subject_brain_region}`-areg
  output_local_areg_hippo_region=${output_delineate_dir}/`basename ${subject_brain_region}`-areg-local
  output_nreg_hippo_region=${output_delineate_dir}/`basename ${subject_brain_region}`-nreg
  output_nreg_template_hippo_image=${output_delineate_dir}/${output_study_id}-${output_series_number}-${output_echo_number}-hippo.img
  output_nreg_template_hippo_dof=${output_delineate_dir}/${output_study_id}-${output_series_number}-${output_echo_number}-hippo.dof
  output_areg_mm_hippo_region=${output_delineate_dir}/`basename ${subject_brain_region}`-areg-mm
  output_nreg_mm_hippo_region=${output_delineate_dir}/`basename ${subject_brain_region}`-nreg-mm
  output_nreg_thresholded_hippo_region=${output_delineate_dir}/`basename ${subject_brain_region}`-nreg-thresholded
  # Delineate the hippo.   
  hippo_delineation ${subject_image} ${subject_brain_region} ${atlas_image} ${atlas_brain_region} \
                    ${output_areg_template_brain_image} ${output_areg_template_brain_region} \
                    ${output_areg_template_brain_series_number} ${output_areg_template_air} \
                    ${atlas_hippo_region} ${output_areg_hippo_region} \
                    ${output_delineate_dir} ${output_study_id} ${output_series_number} ${output_echo_number}  \
                    ${output_local_areg_hippo_region} ${output_nreg_hippo_region} ${output_nreg_template_hippo_image} \
                    ${output_nreg_template_hippo_dof} ${output_areg_mm_hippo_region} ${output_nreg_mm_hippo_region} ${output_nreg_thresholded_hippo_region}
fi     


if [ "${atlas_right_hippo_region}" != "dummy" ] 
then 
  output_delineate_dir=${output_right_dir}
  output_series_number=556
  atlas_hippo_region=${atlas_right_hippo_region}
  
  output_areg_template_brain_image=${output_delineate_dir}/${output_study_id}-${output_areg_template_brain_series_number}-${output_echo_number}.img
  output_areg_template_brain_region=${output_delineate_dir}/`basename ${subject_brain_region}`
  output_areg_template_air=${output_delineate_dir}/${output_study_id}.air
  output_areg_hippo_region=${output_delineate_dir}/`basename ${subject_brain_region}`-areg
  output_local_areg_hippo_region=${output_delineate_dir}/`basename ${subject_brain_region}`-areg-local
  output_nreg_hippo_region=${output_delineate_dir}/`basename ${subject_brain_region}`-nreg
  output_nreg_template_hippo_image=${output_delineate_dir}/${output_study_id}-${output_series_number}-${output_echo_number}-hippo.img
  output_nreg_template_hippo_dof=${output_delineate_dir}/${output_study_id}-${output_series_number}-${output_echo_number}-hippo.dof
  output_areg_mm_hippo_region=${output_delineate_dir}/`basename ${subject_brain_region}`-areg-mm
  output_nreg_mm_hippo_region=${output_delineate_dir}/`basename ${subject_brain_region}`-nreg-mm
  output_nreg_thresholded_hippo_region=${output_delineate_dir}/`basename ${subject_brain_region}`-nreg-thresholded
  # Delineate the hippo.   
  hippo_delineation ${subject_image} ${subject_brain_region} ${atlas_image} ${atlas_brain_region} \
                    ${output_areg_template_brain_image} ${output_areg_template_brain_region} \
                    ${output_areg_template_brain_series_number} ${output_areg_template_air} \
                    ${atlas_hippo_region} ${output_areg_hippo_region} \
                    ${output_delineate_dir} ${output_study_id} ${output_series_number} ${output_echo_number}  \
                    ${output_local_areg_hippo_region} ${output_nreg_hippo_region} ${output_nreg_template_hippo_image} \
                    ${output_nreg_template_hippo_dof} ${output_areg_mm_hippo_region} ${output_nreg_mm_hippo_region} ${output_nreg_thresholded_hippo_region}
fi     






