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
#  Last Changed      : $LastChangedDate: 2011-02-22 10:40:40 +0000 (Tue, 22 Feb 2011) $ 
#  Revision          : $Revision: 5284 $
#  Last modified by  : $Author: kkl $
#
#  Original author   : leung@drc.ion.ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/

#set -x 

source _niftkCommon.sh

# Default values used on img-135 by Kelvin. 
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
  export F3D_BIN=/home/samba/user/leung/work/nifti-reg/nifty_reg-1.2/reg-apps
fi   

script_dir=`dirname $0`

function Usage()
{
cat <<EOF

This script is a wrapper which calls _regAIR-standard-space-all-timepointss.sh to perform registration of all time points to standard space. 

Usage: $0 input_file output_dir 

Mandatory Arguments:
 
  input_dir                : is the directory containing your images and regions. 
  fileContainingImageNames : is a file containing:
  
                             mni_305_image mni_305_region baseline_image baseline_region repeat_image1 repeat_region1 ...

  outputDir                : is where the output is writen to.

EOF
exit 127
}

ndefargs=3

# Check args
if [ $# -lt ${ndefargs} ]; then
  Usage
fi

input_dir=$1
input_file=$2
output_dir=$3

command_filename=regAIR-standard-space-all-timepoints-`date +"%Y%m%d-%H%M%S"`.txt

# Process each line in the input file. 
function iterate_through_input_file
{
  local input_file=$1 
  local do_or_check=$2
  
  cat ${input_file} | while read each_line
  do
    if [ "${each_line}" == "" ]
    then 
      continue
    fi 
  
    echo ${script_dir}/_regAIR-standard-space-all-timepoints.sh \
         ${input_dir} \
         ${output_dir} \
         ${each_line} >> ${command_filename}
  done   
}

# Create output directory. 
mkdir -p ${output_dir}

check_file_exists "${input_file}" "no"
dos_2_unix "${input_file}"

# We then iterate through file, generating commands to a file.
iterate_through_input_file "${input_file}" 1

# Now run the file of commands.
run_batch_job ${command_filename}


































