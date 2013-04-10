#!/bin/bash

#/*============================================================================
#
#  NifTK: A software platform for medical image computing.
#
#  Copyright (c) University College London (UCL). All rights reserved.
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.
#
#  See LICENSE.txt in the top level directory for details.
#
#============================================================================*/

function Usage()
{
cat <<EOF

This script is a working wrapper which is called by MAPS-build-brain-template-library-batch.sh to build a template library. Please use MAPS-build-brain-template-library-batch.sh. 

EOF
exit 127
}

# Check args
if [ $# -lt 3 ]; then
  Usage
fi


# Debug
# set -x

set -u 

export reg_template_loc=${MIDAS_BIN}/reg-template-loc

#
# Script to build a template library. 
# See A comparison of methods for the automated calculation of volumes and atrophy rates in the hippocampus
#     J. Barnes et al, NeuroImage, 2008. 
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

export ffdroitransformation=${MIDAS_FFD}/ffdroitransformation.sh

# Source the common function. 
source MAPS-common.sh
source MAPS-brain-to-brain-registration-without-repeat-mask.sh 

target_atlas=${1}
target_atlas_brain_region=${2}
source_atlas=${3}
source_atlas_brain_region=${4}
source_atlas_vents_region=${5}
output_dir=${6}
only_output_path=${7}

source_atlas_basename=`basename ${source_atlas}`
source_study_id=`echo ${source_atlas_basename} | awk -F- '{printf $1}'`

output_reg_brain_image=${output_dir}/${source_study_id}-library-space.img
output_reg_brain_region=${output_dir}/${source_study_id}-library-space
output_reg_brain_series_number=567
output_reg_air=${output_dir}/${source_study_id}-library-space.air
subject_brain_region=${source_atlas_brain_region}

if [ ${only_output_path} != "yes" ]
then 
  # 12-dof brain-brain registration. 
#brain_to_brain_registration_without_repeat_mask ${target_atlas} ${target_atlas_brain_region} ${source_atlas}  \
#    ${output_reg_brain_image} ${output_reg_brain_series_number} ${output_reg_air} "no"
#regslice ${output_reg_air} ${source_atlas_brain_region} ${output_reg_brain_region} 500
  brain_to_brain_registration_without_repeat_mask_using_irtk ${target_atlas} ${target_atlas_brain_region} ${source_atlas}  \
    ${output_reg_brain_image} ${output_reg_brain_series_number} ${output_reg_air}.dof "no"
  ${ffdroitransformation} ${source_atlas_brain_region} ${output_reg_brain_region} ${source_atlas} ${target_atlas} ${output_reg_air}.dof -bspline
                              
fi                             

output_reg_brain_image_dilated=${output_dir}/${source_study_id}-library-space-dilated.img
output_reg_brain_region_dilated=${output_dir}/${source_study_id}-library-space-dilated
if [ ${only_output_path} != "yes" ]
then 
  # Dilate the region by 2. 
  makemask ${target_atlas} ${output_reg_brain_region} ${output_reg_brain_image_dilated} -d 2                   
  makeroi -img ${output_reg_brain_image_dilated} -out ${output_reg_brain_region_dilated} -alt 128
fi 

patient_id=`imginfo ${source_atlas} -info | awk '{printf $5}'`

echo ${source_study_id}-library-space.img,${source_study_id}-library-space-dilated,${source_study_id}-library-space,${patient_id} >> ${output_dir}/brain-template-library.csv
                            
# Original images.                              
#001_13897,61000-665-1.img,61100-998-1.img,30408-333-1.img,Jen_30408_1011090534,Ash_30408_1099321879,Ash_30408_1107183620,hippoorigleft/30408-333-1.img,hippoorigleft/Jen_30408_1011090534,hippoorigleft/Ash_30408_1099321879,hippoorigright/Ash_30408_1107183620                            
original_library="${source_study_id},${source_study_id}-library-space.img,dummy,dummy,dummy,dummy,dummy"

source_atlas_relative_path=`echo ${source_atlas} | awk -F"${output_dir}" '{printf $2}'`
source_atlas_brain_region_relative_path=`echo ${source_atlas_brain_region} | awk -F"${output_dir}" '{printf $2}'`
source_atlas_vents_region_relative_path=`echo ${source_atlas_vents_region} | awk -F"${output_dir}" '{printf $2}'`

original_library="${original_library},${source_atlas_relative_path},${source_atlas_brain_region_relative_path},dummy,dummy,${source_atlas_vents_region_relative_path}"
 
echo ${original_library} >> ${output_dir}/standard-space-template-library.csv

# midas -img ${output_dir}/${source_study_id}-library-space.img -dual ${target_atlas} -reg ${output_dir}/${source_study_id}-library-space-dilated >/dev/null 2>&1


