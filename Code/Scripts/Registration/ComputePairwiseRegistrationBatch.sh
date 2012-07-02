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

This script is a wrapper which calls ComputeInitialAffineAtlas.sh to perform groupwise affine registration.

Usage: $0 input_file output_dir 

Mandatory Arguments:
 
  input_file : is the input file containing paths to your images and regions. 
                  image1, brain_region1, local_region1a, local_region1b, image2, brain_region2, local_region2a, local_region2b, ......                
                  (just use local_region will be skipped if not found - just specified some dummy name there if no local region)
  output_dir : is the output directory. 
  
Optional arguements:  
  
  -dilation [int]     : number of dilations for the masks [0]. 
  -symmetric [sym_midway/sym/no] : symmetric regsitration option [sym_midway]. 
  -dof [int]          : rigid=2, rigid+scale=3, affine=4 [4]. 
  -ajc [yea/no]       : intensity correction usign the affine scaling factor [no]. 
  -scaling_using_skull [yes/no] : using skull for finding scaling [no]. 
  -similarity [int]   : similarity used in the registration [4]. 
                        1. Sum Squared Difference
                        2. Mean Squared Difference
                        3. Sum Absolute Difference
                        4. Normalized Cross Correlation
                        5. Ratio Image Uniformity
                        6. Partitioned Image Uniformity
                        7. Joint Entropy
                        8. Mutual Information
                        9. Normalized Mutual Information
  -region [yes/no]    : brain region is used or not [yes].
  -pptol [float]      : stopping criteria for the powell optimization. [0.0001]
                          
  

EOF
exit 127
}

ndefargs=2
dilation=0
symmetric="-sym_midway"
scaling_using_skull="no"
dof=4
similarity=4
ajc="no"
region="yes"
pptol=0.0001

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
    -dilation)
      dilation=$2
      shift 1
      ;;      
    -symmetric)
      symmetric=$2
      shift 1
      ;;
    -dof)
      dof=$2
      shift 1
      ;;
    -scaling_using_skull)
      scaling_using_skull=$2
      shift 1
      ;;
    -similarity)
      similarity=$2
      shift 1
      ;;
    -ajc)
      ajc=$2
      shift 1
      ;;
    -region)
      region=$2
      shift 1
      ;;
    -pptol)
      pptol=$2
      shift 1
      ;;
    -*)
        Usage
      exitprog "Error: option $1 not recognised" 1
      ;;
    esac
    shift 1
done


command_filename_tmp=ComputeInitialAffineAtlas-`date +"%Y%m%d-%H%M%S"`.XXXXXXXXXX
command_filename=`mktemp ${command_filename_tmp}`

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
  
    local fixed_image=`echo ${each_line} | awk '{print $1}'`
    local fixed_image_mask=`echo ${each_line} | awk '{print $2}'`
    local moving_images_and_masks=`echo ${each_line} | awk '{ for (i=3; i<=NF; i+=1) { printf $i" " } } '`
    local image_basename=`basename ${fixed_image}`
    local study_number=`echo ${image_basename} | awk -F- '{print $1}'`
    local output_format="${study_number}_pairwise_%i_%i"
    
    if [ ${do_or_check} == 1 ] 
    then 
      echo ${script_dir}/ComputePairwiseRegistration.sh \
           ${output_dir}/${output_format} \
            ${dilation} ${symmetric}  \
           ${dof} ${scaling_using_skull} ${similarity} ${ajc} ${pptol} ${region} \
           ${fixed_image} ${fixed_image_mask} ${moving_images_and_masks}  >> ${command_filename}
    else
      check_file_exists ${image} "no"
      check_file_exists ${image%.img}.hdr "no"
      check_file_exists ${region} "no"
    fi 
  done   
}

# Create output directory. 
mkdir -p ${output_dir}

check_file_exists ${input_file} "no"
dos_2_unix ${input_file}

# We first simply scan through file, cos then we can stop early if there are missing files
iterate_through_input_file ${input_file} 0

# We then iterate through file, generating commands to a file.
iterate_through_input_file ${input_file} 1

# Now run the file of commands.
run_batch_job ${command_filename}


































