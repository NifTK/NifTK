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

This script is a wrapper which calls Hippo-MAPS.sh to perform automated hippocampal segmentation.

Usage: $0 input_file output_dir 

Mandatory Arguments:
 
  input_file : is the input file containing paths to your images and regions. 
  output_dir : is the output directory. 
  
Optional arguements:  

  -mean_brain_intensity : specifiy the mean brain intensity to use [default: from the brain region]. 
  -remove_dir           : remove directories containing intermediate results. 
  -leave_one_out        : do leave one out experiment by skipping the same image name. 
  -library              : option to specify the location of template library index. 
  -threshold (no/upper/lower/both) : specify to use no, upper, lower threshold or both threshold [default: both]. 
  -staple_count a b     : number of segmentations (from a to b) to combine [default: 8 8]. 
  -only_staple          : only combine the results and do not do any registration. 
  -skip_side (no/left/right) : skip process the no, left or right side. 

EOF
exit 127
}

# Check args
if [ $# -lt 3 ]; then
  Usage
fi


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
# F3D not used for now. 
export F3D_BIN=/home/samba/user/leung/work/nifti-reg/nifty_reg-1.2/reg-apps
export HIPPO_TEMPLATE_LIBRARY=/var/drc/software/32bit/niftk-data/hippo-template-library

# set up shell for the cluster jobs. 
export SGE_SHELL=/bin/bash 

script_dir=`dirname $0`


ndefargs=2
do_threshold="both"
mean_brain_intensity=-1
remove_dir="no"
leave_one_out="no"
staple_count=8
staple_count_start=8
staple_count_end=8
only_staple="no"
process_left="yes"
process_right="yes"

# Check args
if [ $# -lt ${ndefargs} ]; then
  Usage
fi

input_file=$1
output_dir=$2

# Parse remaining command line options
shift $ndefargs
while [ "$#" -gt 0 ]
do
    case $1 in
      -threshold)
        do_threshold=$2
        shift 1
      ;;
     -remove_dir)
        remove_dir="yes"
      ;;
     -leave_one_out)
        leave_one_out="yes"
      ;;
     -mean_brain_intensity)
        mean_brain_intensity=$2
        shift 1
      ;;
     -library)
        HIPPO_TEMPLATE_LIBRARY=$2
        shift 1
      ;; 
     -staple_count)
        staple_count_start=$2
        shift 1
        staple_count_end=$2
        shift 1
      ;;
     -only_staple)
        only_staple="yes"
      ;;
     -skip_side)
        if [ "$2" == "left" ]
        then
          process_left="no"
        elif  [ "$2" == "right" ]
        then 
          process_right="no"
        fi 
        shift 1
      ;;
     -*)
        Usage
        exitprog "Error: option $1 not recognised" 1
      ;;
    esac
    shift 1
done


command_filename=MAPS-hippo-`date +"%Y%m%d-%H%M%S"`.txt

# Process each line in the input file. 
function iterate_through_input_file
{
  local input_file=$1 
  local do_or_check=$2
  
  local mrf_weighting=0.2
  
  cat ${input_file} | while read each_line
  do
    local image=`echo ${each_line} | awk '{print $1}'`
    local region=`echo ${each_line} | awk '{print $2}'`
    
    local image_basename=`basename ${image}`
    local study_number=`echo ${image_basename} | awk -F- '{print $1}'`
    
    if [ ${do_or_check} == 1 ] 
    then 
      echo ${script_dir}/Hippo-MAPS.sh \
          ${HIPPO_TEMPLATE_LIBRARY}  \
          ${image} \
          ${region} \
          ${study_number} \
          ${output_dir}/hippo-left \
          ${output_dir}/hippo-right ${staple_count} ${mrf_weighting} \
          ${do_threshold} ${mean_brain_intensity} ${remove_dir} ${leave_one_out} ${only_staple} \
          ${process_left} ${process_right} ${staple_count_start} ${staple_count_end} >> ${command_filename}
    else
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


