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
#  Last Changed      : $LastChangedDate: 2011-07-14 13:44:59 +0100 (Thu, 14 Jul 2011) $ 
#  Revision          : $Revision: 6750 $
#  Last modified by  : $Author: kkl $
#
#  Original author   : leung@drc.ion.ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/

source _niftkCommon.sh

export MIDAS_BIN=/var/NOT_BACKUP/work/midas/build/bin
export FFITK_BIN=/var/NOT_BACKUP/work/ffitk/bin
export MIDAS_FFD=/var/NOT_BACKUP/work/midasffd
export FSLDIR=/var/lib/midas/pkg/i686-pc-linux-gnu/fsl
export F3D_BIN=/home/samba/user/leung/work/nifti-reg/nifty_reg-1.2/reg-apps

script_dir=`dirname $0`

#
# Batch script to build a template library. 
# See A comparison of methods for the automated calculation of volumes and atrophy rates in the hippocampus
#     J. Barnes et al, NeuroImage, 2008. 
#

function Usage()
{
cat <<EOF

This script is a wrapper which calls MAPS-build-template-library.sh to build a template library.

Usage: $0 input_file output_dir 

Mandatory Arguments:
 
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


input_file=$1
output_dir=$2
output_file_only=$3

command_filename=MAPS-build-template-library-`date +"%Y%m%d-%H%M%S"`.txt

# Process each line in the input file. 
function iterate_through_input_file
{
  local input_file=$1 
  local do_or_check=$2
  
  cat ${input_file} | while read each_line
  do
    local target_atlas=`echo ${each_line} | awk -F, '{print $1}'`
    local target_atlas_brain_region=`echo ${each_line} | awk -F, '{print $2}'`
    local target_left_local_region=`echo ${each_line} | awk -F, '{print $3}'`
    local target_right_local_region=`echo ${each_line} | awk -F, '{print $4}'`
    local source_atlas=`echo ${each_line} | awk -F, '{print $5}'`
    local source_atlas_brain_region=`echo ${each_line} | awk -F, '{print $6}'`
    local source_left_local_region=`echo ${each_line} | awk -F, '{print $7}'`
    local source_right_local_region=`echo ${each_line} | awk -F, '{print $8}'`
    
    if [ ${do_or_check} == 1 ] 
    then 
      echo ${script_dir}/MAPS-build-template-library.sh \
        ${target_atlas} ${target_atlas_brain_region} \
        ${target_left_local_region} ${target_right_local_region} \
        ${source_atlas} ${source_atlas_brain_region} \
        ${source_left_local_region} ${source_right_local_region} \
        ${output_dir} ${output_file_only} >> ${command_filename}
        
      # Should put the target atlas outside this loop.....
      echo "left-template-library.csv,right-template-library.csv,standard-space-template-library.csv,`basename ${target_atlas}`,`basename ${target_atlas_brain_region}`,`basename ${target_left_local_region}`,`basename ${target_right_local_region}`" > ${output_dir}/index
      
    else
      check_file_exists ${target_atlas} "no"
      check_file_exists ${target_atlas_brain_region} "no"
      check_file_exists ${target_left_local_region} "no"
      check_file_exists ${target_right_local_region} "no"
      check_file_exists ${source_atlas} "no"
      check_file_exists ${source_atlas_brain_region} "no"
      if [ "${source_left_local_region}" != "dummy" ] 
      then 
        check_file_exists ${source_left_local_region} "no"
      fi 
      if [ "${source_right_local_region}" != "dummy" ] 
      then 
        check_file_exists ${source_right_local_region} "no"
      fi 
    fi 
  done   
}

# Create output directory. 
mkdir -p ${output_dir}

check_file_exists ${input_file} "no"
dos_2_unix ${input_file}

rm -f ${output_dir}/standard-space-template-library.csv ${output_dir}/left-template-library.csv ${output_dir}/right-template-library.csv index

# We first simply scan through file, cos then we can stop early if there are missing files
iterate_through_input_file ${input_file} 0

# We then iterate through file, generating commands to a file.
iterate_through_input_file ${input_file} 1

# Now run the file of commands.
run_batch_job ${command_filename}



