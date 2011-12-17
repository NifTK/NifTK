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
#  Last Changed      : $LastChangedDate: 2010-06-11 14:40:42 +0100 (Fri, 11 Jun 2010) $ 
#  Revision          : $Revision: 3375 $
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
  # F3D not used for now. 
  export F3D_BIN=/home/samba/user/leung/work/nifti-reg/nifty_reg-1.2/reg-apps
fi   

script_dir=`dirname $0`

function Usage()
{
cat <<EOF

This script is a wrapper which calls Hippo-MAPS.sh to perform automated hippocampal segmentation.

Usage: $0 input_file output_dir 

Mandatory Arguments:
 
  input_file : is the input file containing paths to your images and regions. 
  output_dir : is the output directory. 
  
Optional arguements:  

  -staple_count     : number of segmentations to STAPLE [8]. 
  -mrf_weighting    : MRF weight after STAPLE [0.2]. 
  -template_library : location of the template library [/var/drc/software/32bit/niftk-data/hippo-template-library]. 
  -nreg             : non-rigid registration to be (f3d or ffd) [f3d]. 
  -cc               : connected components after threshold [no]. 
  -threshold_range  : threshold the propagated region [0.7 1.1]. 


EOF
exit 127
}

ndefargs=2
staple_count=8
mrf_weighting=0.2
template_library=/var/drc/software/32bit/niftk-data/hippo-template-library
nreg=f3d
cc=no
threshold_lower=0.7
threshold_upper=1.1


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
      -staple_count)
        staple_count=$2
        shift 1
      ;;
     -mrf_weighting)
        mrf_weighting=$2
        shift 1
      ;;
     -template_library)
        template_library=$2
        shift 1
      ;;
     -nreg)
        nreg=$2
        shift 1
      ;;
     -cc)
       cc=$2
       shift 1
      ;;
     -threshold_range)
       threshold_lower=$2
       threshold_upper=$3
       shift 2
      ;;
     -*)
        Usage
        exitprog "Error: option $1 not recognised" 1
      ;;
    esac
    shift 1
done

# Index should contain the details of the template library. 
# watjo refers to the reference image. 
index=`cat ${template_library}/index`
left_hippo_template_library=${template_library}/`echo ${index}| awk -F, '{printf $1}'`
right_hippo_template_library=${template_library}/`echo ${index}| awk -F, '{printf $2}'`
hippo_template_library_original=${template_library}/`echo ${index}| awk -F, '{printf $3}'`
watjo_image=${template_library}/`echo ${index}| awk -F, '{printf $4}'`
watjo_brain_region=${template_library}/`echo ${index}| awk -F, '{printf $5}'`
watjo_hippo_left_region=${template_library}/`echo ${index}| awk -F, '{printf $6}'`
watjo_hippo_right_region=${template_library}/`echo ${index}| awk -F, '{printf $7}'`

command_filename=MAPS-generic-`date +"%Y%m%d-%H%M%S"`.txt

# Process each line in the input file. 
function iterate_through_input_file
{
  local input_file=$1 
  local do_or_check=$2
  
  cat ${input_file} | while read each_line
  do
    local image=`echo ${each_line} | awk '{print $1}'`
    local region=`echo ${each_line} | awk '{print $2}'`
    local template_subject_id=`echo ${each_line} | awk '{print $3}'`
    
    local image_basename=`basename ${image}`
    local study_number=`echo ${image_basename} | awk -F- '{print $1}'`
    
    if [ ${do_or_check} == 1 ] 
    then 
      echo ${script_dir}/Generic-MAPS.sh \
          ${template_library}  \
          ${image} \
          ${region} \
          ${study_number} \
          ${output_dir}/left \
          ${output_dir}/right ${staple_count} ${mrf_weighting} \
          ${left_hippo_template_library} \
          ${right_hippo_template_library} \
          ${hippo_template_library_original} \
          ${watjo_image} \
          ${watjo_brain_region} \
          ${watjo_hippo_left_region} \
          ${watjo_hippo_right_region} ${nreg} ${cc} ${threshold_lower} ${threshold_upper} ${template_subject_id} >> ${command_filename}
    else
      check_file_exists ${image} "no"
      check_file_exists ${image%.img}.hdr "no"
      check_file_exists ${region} "no"
    fi 
  done   
}

# Create output directory. 
mkdir -p ${output_dir}/left
mkdir -p ${output_dir}/right

check_file_exists ${input_file} "no"
dos_2_unix ${input_file}

# We first simply scan through file, cos then we can stop early if there are missing files
iterate_through_input_file ${input_file} 0

# We then iterate through file, generating commands to a file.
iterate_through_input_file ${input_file} 1

# Now run the file of commands.
run_batch_job ${command_filename}


