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
#  Last Changed      : $LastChangedDate: 2011-08-15 10:46:08 +0100 (Mon, 15 Aug 2011) $ 
#  Revision          : $Revision: 7074 $
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

This script is a working wrapper which is called by MAPS-build-template-library-batch.sh to build a template library. Please use MAPS-build-template-library-batch.sh. 

EOF
exit 127
}

# Check args
if [ $# -lt 3 ]; then
  Usage
fi


# Debug
set -x

set -u 

export FFITK_BIN=/var/NOT_BACKUP/work/ffitk/bin
export MIDAS_FFD=/var/NOT_BACKUP/work/midasffd
export FSLDIR=/var/lib/midas/pkg/i686-pc-linux-gnu/fsl

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

# Source the common function. 
source MAPS-common.sh

target_atlas=${1}
target_atlas_brain_region=${2}
target_left_local_region=${3}
target_right_local_region=${4}
source_atlas=${5}
source_atlas_brain_region=${6}
source_left_local_region=${7}
source_right_local_region=${8}
output_dir=${9}
only_output_path=${10}

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
#brain_to_brain_registration ${target_atlas} ${target_atlas_brain_region} ${source_atlas} ${source_atlas_brain_region} \
#                              ${output_reg_brain_image} ${output_reg_brain_region} ${output_reg_brain_series_number} \
#                              ${output_reg_air}
echo already done
fi                             
                            
# Original images.                              
#001_13897,61000-665-1.img,61100-998-1.img,30408-333-1.img,Jen_30408_1011090534,Ash_30408_1099321879,Ash_30408_1107183620,hippoorigleft/30408-333-1.img,hippoorigleft/Jen_30408_1011090534,hippoorigleft/Ash_30408_1099321879,hippoorigright/Ash_30408_1107183620                            
original_library="${source_study_id}"

output_series_number=555
output_echo_number=1
if [ -f "${source_left_local_region}" ]
then 
  # left local registration. 
  left_or_right=left
  eval target_local_region="target_${left_or_right}_local_region"
  eval output_series_number="output_${left_or_right}_series_number"
  output_local_dir=${output_dir}/${left_or_right}
  mkdir -p ${output_local_dir}
  chmod g+w ${output_local_dir}
  output_series_number=555
  output_echo_number=1
  
  if [ ${only_output_path} != "yes" ]
  then 
    # 6-dof hippo-hippo registration. 
    ${reg_template_loc} ${target_atlas} ${output_reg_brain_image} ${target_atlas_brain_region} ${output_reg_brain_region} \
                        ${!target_local_region} dummy \
                        ${output_local_dir} ${output_local_dir} ${output_local_dir} ${output_local_dir} \
                        ${source_study_id} ${output_series_number} ${output_echo_number} 0.5 0.8 \
                        no ${source_study_id}_xor.roi 4 1 1 no 6 6
  fi                     
                      
  eval source_local_region="source_${left_or_right}_local_region"
  output_areg_local_region=${output_local_dir}/${source_study_id}_areg_local
  output_local_areg_local_region=${output_local_dir}/${source_study_id}_areg_local_local
  output_local_areg_local_image_dilated=${output_local_dir}/${source_study_id}_areg_local_local_dilated.img
  output_local_areg_local_region_dilated=${output_local_dir}/${source_study_id}_areg_local_local_dilated
  output_local_areg_source_image=${output_local_dir}/${source_study_id}-${output_series_number}-${output_echo_number}.img
  if [ ${only_output_path} != "yes" ]
  then 
    # for some reason, the output from reg-template-loc is not in the same orientation as the input. 
    anchange ${output_local_areg_source_image} ${output_local_areg_source_image} -setorient cor
    # Reslice local after brain-brain registration.                     
    regslice ${output_reg_air} ${!source_local_region} ${output_areg_local_region} 500 -i 2
    # Reslice after the local 6-dof registration 
    regslice ${output_local_dir}/100001-${source_study_id}.air ${output_areg_local_region} ${output_local_areg_local_region} 500 -i 2
    # Dilate the region by 2. 
    makemask ${target_atlas} ${output_local_areg_local_region} ${output_local_areg_local_image_dilated} -d 2                   
    makeroi -img ${output_local_areg_local_image_dilated} -out ${output_local_areg_local_region_dilated} -alt 128
  fi 
  
  patient_id=`imginfo ${source_atlas} -info | awk '{printf $5}'`
  
  echo "${left_or_right}/${source_study_id}-${output_series_number}-${output_echo_number}.img,${left_or_right}/${source_study_id}_areg_local_local_dilated,${patient_id}" >> ${output_dir}/${left_or_right}-template-library.csv
  
fi   

# Original images.                              
#001_13897,61000-665-1.img,61100-998-1.img,30408-333-1.img,Jen_30408_1011090534,Ash_30408_1099321879,Ash_30408_1107183620,hippoorigleft/30408-333-1.img,hippoorigleft/Jen_30408_1011090534,hippoorigleft/Ash_30408_1099321879,hippoorigright/Ash_30408_1107183620                            
original_library="${original_library},${source_study_id}-${output_series_number}-${output_echo_number}.img"

output_series_number=556
output_echo_number=1

if [ -f "${source_right_local_region}" ]
then 
  # right local registration. 
  left_or_right=right
  eval target_local_region="target_${left_or_right}_local_region"
  eval output_series_number="output_${left_or_right}_series_number"
  output_local_dir=${output_dir}/${left_or_right}
  mkdir -p ${output_local_dir}
  chmod g+w ${output_local_dir}
  output_series_number=556
  output_echo_number=1
  
  if [ ${only_output_path} != "yes" ]
  then 
    # 6-dof hippo-hippo registration. 
    ${reg_template_loc} ${target_atlas} ${output_reg_brain_image} ${target_atlas_brain_region} ${output_reg_brain_region} \
                        ${!target_local_region} dummy \
                        ${output_local_dir} ${output_local_dir} ${output_local_dir} ${output_local_dir} \
                        ${source_study_id} ${output_series_number} ${output_echo_number} 0.5 0.8 \
                        no ${source_study_id}_xor.roi 4 1 1 no 6 6
  fi 
  
  eval source_local_region="source_${left_or_right}_local_region"
  output_areg_local_region=${output_local_dir}/${source_study_id}_areg_local
  output_local_areg_local_region=${output_local_dir}/${source_study_id}_areg_local_local
  output_local_areg_local_image_dilated=${output_local_dir}/${source_study_id}_areg_local_local_dilated.img
  output_local_areg_local_region_dilated=${output_local_dir}/${source_study_id}_areg_local_local_dilated
  output_local_areg_source_image=${output_local_dir}/${source_study_id}-${output_series_number}-${output_echo_number}.img
  if [ ${only_output_path} != "yes" ]
  then 
    # for some reason, the output from reg-template-loc is not in the same orientation as the input. 
    anchange ${output_local_areg_source_image} ${output_local_areg_source_image} -setorient cor
    # Reslice local after brain-brain registration.                     
    regslice ${output_reg_air} ${!source_local_region} ${output_areg_local_region} 500 -i 2
    # Reslice after the local 6-dof registration 
    regslice ${output_local_dir}/100001-${source_study_id}.air ${output_areg_local_region} ${output_local_areg_local_region} 500 -i 2
    # Dilate the region by 2. 
    makemask ${target_atlas} ${output_local_areg_local_region} ${output_local_areg_local_image_dilated} -d 2                   
    makeroi -img ${output_local_areg_local_image_dilated} -out ${output_local_areg_local_region_dilated} -alt 128
  fi   
  

  patient_id=`imginfo ${source_atlas} -info | awk '{printf $5}'`

  echo "${left_or_right}/${source_study_id}-${output_series_number}-${output_echo_number}.img,${left_or_right}/${source_study_id}_areg_local_local_dilated,${patient_id}" >> ${output_dir}/${left_or_right}-template-library.csv
fi   


# Original images.                              
#001_13897,61000-665-1.img,61100-998-1.img,30408-333-1.img,Jen_30408_1011090534,Ash_30408_1099321879,Ash_30408_1107183620,hippoorigleft/30408-333-1.img,hippoorigleft/Jen_30408_1011090534,hippoorigleft/Ash_30408_1099321879,hippoorigright/Ash_30408_1107183620                            
original_library="${original_library},${source_study_id}-${output_series_number}-${output_echo_number}.img"

source_atlas_relative_path=`echo ${source_atlas} | awk -F"${output_dir}" '{printf $2}'`
source_atlas_brain_region_relative_path=`echo ${source_atlas_brain_region} | awk -F"${output_dir}" '{printf $2}'`
source_left_local_region_relative_path=`echo ${source_left_local_region} | awk -F"${output_dir}" '{printf $2}'`
source_right_local_region_relative_path=`echo ${source_right_local_region} | awk -F"${output_dir}" '{printf $2}'`
original_library="${original_library},dummy,dummy,dummy,dummy,${source_atlas_relative_path},${source_atlas_brain_region_relative_path},${source_left_local_region_relative_path},${source_right_local_region_relative_path}"

echo ${original_library} >> ${output_dir}/standard-space-template-library.csv








