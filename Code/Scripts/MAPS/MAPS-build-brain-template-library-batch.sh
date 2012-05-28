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
#  Last Changed      : $LastChangedDate: 2010-06-29 10:08:03 +0100 (Tue, 29 Jun 2010) $ 
#  Revision          : $Revision: 3439 $
#  Last modified by  : $Author: kkl $
#
#  Original author   : leung@drc.ion.ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/

# set -x

source _niftkCommon.sh

if [ ! -n "${MIDAS_BIN}" ]
then 
  export MIDAS_BIN=/var/NOT_BACKUP/work/midas/build/bin
fi 
if [ ! -n "${FFITK_BIN}" ]
then
  export FFITK_BIN=/var/NOT_BACKUP/work/ffitk/bin
fi 
if [ ! -n "${MIDAS_FFD}" ]
then
  export MIDAS_FFD=/var/NOT_BACKUP/work/midasffd
fi   
if [ ! -n "${FSLDIR}" ]
then 
  export FSLDIR=/var/lib/midas/pkg/i686-pc-linux-gnu/fsl
fi   
if [ ! -n "${F3D_BIN}" ]
then 
  # F3D not used for now. 
  export F3D_BIN=/home/samba/user/leung/work/nifti-reg/nifty_reg-1.2/reg-apps
fi   


script_dir=`dirname $0`

#
# Batch script to build a template library. 
# See A comparison of methods for the automated calculation of volumes and atrophy rates in the hippocampus
#     J. Barnes et al, NeuroImage, 2008. 
#

function Usage()
{
cat <<EOF

This script is a wrapper which calls MAPS-build-brain-template-library.sh to build a brain template library.

Usage: $0 input_file output_dir 

Mandatory Arguments:

  target_atlas : average image as the target image to be registered to. 
  
  target_atlas_brain_region : brain region of the target image. 
 
  input_file : is the input file containing paths to your images and regions, in the format below: 
    target_atlas, target_atlas_brain_region, target_left_local_region, target_right_local_region, source_atlas, 
    source_atlas_brain_region, source_left_local_region, source_right_local_region
  
  output_dir : is the output directory. 
  
  output_file_only : only output the library text file ("yes" or "no"). 

EOF
exit 127
}

# Check args
if [ $# -lt 3 ]; then
  Usage
fi


target_atlas=$1
target_atlas_brain_region=$2
input_file=$3
output_dir=$4
output_file_only=$5

command_filename=MAPS-build-brain-template-library-`date +"%Y%m%d-%H%M%S"`.txt

# Process each line in the input file. 
function iterate_through_input_file
{
  local input_file=$1 
  local do_or_check=$2
  local target_atlas=$3
  local target_atlas_brain_region=$4
  
  cat ${input_file} | while read each_line
  do
    local source_atlas=`echo ${each_line} | awk -F, '{print $1}'`
    local source_atlas_brain_region=`echo ${each_line} | awk -F, '{print $2}'`
    local source_atlas_vents_region=`echo ${each_line} | awk -F, '{print $3}'`
    
    if [ ${do_or_check} == 1 ] 
    then 
      echo ${script_dir}/MAPS-build-brain-template-library.sh \
        ${target_atlas} ${target_atlas_brain_region} \
        ${source_atlas} ${source_atlas_brain_region} ${source_atlas_vents_region} \
        ${output_dir} ${output_file_only} >> ${command_filename}
        
      # Should put the target atlas outside this loop.....
      
    else
      check_file_exists ${target_atlas} "no"
      check_file_exists ${target_atlas_brain_region} "no"
      check_file_exists ${source_atlas} "no"
      check_file_exists ${source_atlas_brain_region} "no"
      check_file_exists ${source_atlas_vents_region} "no"
    fi 
  done   
}

# Create output directory. 
mkdir -p ${output_dir}

check_file_exists ${input_file} "no"
dos_2_unix ${input_file}

rm -f ${output_dir}/brain-template-library.csv ${output_dir}/standard-space-template-library.csv

# We first simply scan through file, cos then we can stop early if there are missing files
iterate_through_input_file ${input_file} 0 ${target_atlas} ${target_atlas_brain_region}

# We then iterate through file, generating commands to a file.
iterate_through_input_file ${input_file} 1 ${target_atlas} ${target_atlas_brain_region}

# Now run the file of commands.
run_batch_job ${command_filename}

echo "brain-template-library.csv,dummy,standard-space-template-library.csv,`basename ${target_atlas}`,`basename ${target_atlas_brain_region}`" > ${output_dir}/index




