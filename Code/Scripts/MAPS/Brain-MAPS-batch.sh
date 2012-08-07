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
#  Last Changed      : $LastChangedDate: 2011-09-05 09:56:45 +0100 (Mon, 05 Sep 2011) $ 
#  Revision          : $Revision: 7234 $
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
  export F3D_BIN=/home/samba/user/leung/work/nifti-reg/nifty_reg-1.2/reg-apps
fi   

# set up shell for the cluster jobs. 
export SGE_SHELL=/bin/bash 

script_dir=`dirname $0`

function Usage()
{
cat <<EOF

This script is a wrapper which calls Brain-MAPS.sh to perform automated brain segmentation. The input image should be in sagittal slices, unless "-use_orientation yes" is specified for Midas images. 

Usage: $0 input_file output_dir 

Mandatory Arguments:
 
  input_file : is the input file containing paths to your images and regions. 
  output_dir : is the output directory. 
  
Optional arguements:  

  -staple_counts    : a range of number of segmentations to STAPLE [8 8]. 
  -mrf_weighting    : MRF weight after STAPLE [0.2]. 
  -template_library : location of the template library [/var/drc/software/32bit/niftk-data/hippo-template-library]. 
  -dilation_for_f3d : dilation of the mask before F3D nreg [4]. 
  -nreg             : nreg to use (f3d or ffd) [f3d]. 
  -f3d_brain_prereg : f3d to use whole image pre-registration (yes or no) [no]. 
  -areg             : areg to use (air or irtk_areg) [air]. 
  -cpp              : control point spacing [5]. 
  -f3d_energy       : bending energy for f3d [0.01].
  -f3d_iterations   : number of iterations in f3d [300]. 
  -confidence       : confidence used in the STAPLE [0.5]. 
  -vents_or_not     : include vents in the brain segmentation [no]. 
  -remove_dir       : remove results directories to save space [no]. 
  -use_orientation  : use the orientation flag in Midas [no]. 
  -leaveoneout      : apply the leave-one-out test [yes].
  -kmeans           : use kmeans-clustering to determine intensity for condition dilation [no]. 
  -init_9dof        : use 9dof for the global reg initialisatino. 
  -cd_mode          : conditional dilation mode [2]. 

EOF
exit 127
}

ndefargs=2
staple_count_start=8
staple_count_end=8
mrf_weighting=0.2
dilation_for_f3d=4
nreg=f3d
template_library=/var/drc/software/32bit/niftk-data/hippo-template-library
f3d_brain_prereg=no
areg=air
cpp=5
f3d_energy=0.01
f3d_iterations=300
confidence=0.5
vents_or_not=no
remove_dir=no
use_orientation=no
leaveoneout=yes
kmeans=no
init_9dof=no
cd_mode=2

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
      -staple_counts)
        staple_count_start=$2
        staple_count_end=$3
        shift 2
      ;;
     -mrf_weighting)
        mrf_weighting=$2
        shift 1
      ;;
     -template_library)
        template_library=$2
        shift 1
      ;;
     -dilation_for_f3d)
        dilation_for_f3d=$2
        shift 1
      ;;
     -nreg)
        nreg=$2
        shift 1
      ;;
     -f3d_brain_prereg)
        f3d_brain_prereg=$2
        shift 1
      ;;
     -areg)
        areg=$2
        shift 1
      ;;
     -cpp)
        cpp=$2
        shift 1
      ;;
     -f3d_energy)
        f3d_energy=$2
        shift 1
      ;;
     -f3d_iterations)
        f3d_iterations=$2
        shift 1
      ;;
     -confidence)
        confidence=$2
        shift 1
      ;;
     -vents_or_not)
        vents_or_not=$2
        shift 1
      ;;
     -remove_dir)
        remove_dir=$2
        shift 1
      ;;
     -use_orientation)
        use_orientation=$2
        shift 1
      ;;
     -leaveoneout)
        leaveoneout=$2
        shift 1
      ;;
     -kmeans)
        kmeans=$2
        shift 1
      ;;
     -init_9dof)
        init_9dof=$2
        shift 1
      ;;
     -cd_mode)
        cd_mode=$2
        shift 1
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

temp_name=MAPS-generic-`date +"%Y%m%d-%H%M%S"`.XXXXXXXXXX
command_filename=`mktemp ${temp_name}`

# Process each line in the input file. 
function iterate_through_input_file
{
  local input_file=$1 
  local do_or_check=$2
  
  cat ${input_file} | while read each_line
  do
    local image=`echo ${each_line} | awk '{print $1}'`
    
    local image_basename=`basename ${image}`
    local study_number=`echo ${image_basename} | awk -F- '{print $1}'`
    
    if [ ${do_or_check} == 1 ] 
    then 
      echo ${script_dir}/Brain-MAPS.sh \
          ${template_library}  \
          ${image} \
          ${study_number} \
          ${output_dir}/left \
          ${staple_count_start} ${staple_count_end} ${mrf_weighting} \
          ${left_hippo_template_library} \
          ${hippo_template_library_original} \
          ${watjo_image} \
          ${watjo_brain_region} ${dilation_for_f3d} ${nreg} ${f3d_brain_prereg} \
          ${areg} ${cpp} ${f3d_energy} ${f3d_iterations} \
          ${confidence} ${vents_or_not} ${remove_dir} ${use_orientation} \
          ${leaveoneout} ${kmeans} ${init_9dof} ${cd_mode} >> ${command_filename}
    else
      check_file_exists ${image} "no"
      check_file_exists ${image%.img}.hdr "no"
      check_file_exists ${region} "no"
    fi 
  done   
}

# Create output directory. 
mkdir -p ${output_dir}/left

check_file_exists ${input_file} "no"
dos_2_unix ${input_file}

# We first simply scan through file, cos then we can stop early if there are missing files
iterate_through_input_file ${input_file} 0

# We then iterate through file, generating commands to a file.
iterate_through_input_file ${input_file} 1

# Now run the file of commands.
run_batch_job ${command_filename}


