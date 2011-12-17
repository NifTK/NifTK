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

source _niftkCommon.sh

if [ "${MIDAS_BIN}" == "" ] 
then 
  export MIDAS_BIN=/var/NOT_BACKUP/work/midas/build/bin
fi 
if [ "${FFITK_BIN}" == "" ] 
then 
  export FFITK_BIN=/var/NOT_BACKUP/work/ffitk/bin
fi 
if [ "${MIDAS_FFD}" == "" ] 
then 
  export MIDAS_FFD=/var/NOT_BACKUP/work/midasffd
fi 
if [ "${FSLDIR}" == "" ] 
then 
  export FSLDIR=/var/lib/midas/pkg/i686-pc-linux-gnu/fsl
fi 
if [ "${F3D_BIN}" == "" ] 
then 
  export F3D_BIN=/home/samba/user/leung/work/nifti-reg/nifty_reg-1.2/reg-apps
fi   

script_dir=`dirname $0`

function Usage()
{
cat <<EOF

This script is a wrapper which calls Hippo-Propagate.sh to propagate the hippocampal segmentation from one image to another.

Usage: $0 input_file output_dir 

Mandatory Arguments:
 
  input_file : is the input file containing paths to your images and regions. 
               atlas_image, atlas_brain_region, atlas_left_hippo_region, atlas_right_hippo_region, subject_image, subject_brain_region
               
  output_dir : is the output directory. 

EOF
exit 127
}

# Check args
if [ $# -lt 2 ]; then
  Usage
fi


input_file=$1
output_dir=$2

command_filename=MAPS-Propagate-`date +"%Y%m%d-%H%M%S"`.txt

# Process each line in the input file. 
function iterate_through_input_file
{
  local input_file=$1 
  local do_or_check=$2
  
  cat ${input_file} | while read each_line
  do
    local atlas_image=`echo ${each_line} | awk -F, '{print $1}'`
    local atlas_brain_region=`echo ${each_line} | awk -F, '{print $2}'`
    local atlas_left_hippo_region=`echo ${each_line} | awk -F, '{print $3}'`
    local atlas_right_hippo_region=`echo ${each_line} | awk -F, '{print $4}'`
    local image=`echo ${each_line} | awk -F, '{print $5}'`
    local region=`echo ${each_line} | awk -F, '{print $6}'`
    
    local image_basename=`basename ${image}`
    local study_number=`echo ${image_basename} | awk -F- '{print $1}'`
    
    if [ ${do_or_check} == 1 ] 
    then 
      echo ${script_dir}/Hippo-Propagate.sh \
          ${atlas_image} \
          ${atlas_brain_region} \
          ${atlas_left_hippo_region} \
          ${atlas_right_hippo_region} \
          ${image} \
          ${region} \
          ${study_number} \
          ${output_dir}/hippo-left \
          ${output_dir}/hippo-right >> ${command_filename}
    else
      check_file_exists ${atlas_image} "no"
      check_file_exists ${atlas_brain_region} "no"
      check_file_exists ${atlas_left_hippo_region} "no"
      check_file_exists ${atlas_right_hippo_region} "no"
      check_file_exists ${image} "no"
      check_file_exists ${image%.img}.hdr "no"
      check_file_exists ${region} "no"
    fi 
  done   
}

# Create output directory. 
mkdir -p ${output_dir}/hippo-left
mkdir -p ${output_dir}/hippo-right

check_file_exists ${input_file} "no"
dos_2_unix ${input_file}

# We first simply scan through file, cos then we can stop early if there are missing files
iterate_through_input_file ${input_file} 0

# We then iterate through file, generating commands to a file.
iterate_through_input_file ${input_file} 1

# Now run the file of commands.
run_batch_job ${command_filename}


