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
set -x

current_dir=`dirname $0`

input_dir=$1
output_dir=$2
ss_image=$3
ss_region=$4
baseline_image=$5
baseline_region=$6
starting_arg=7

check_file_exists "${input_dir}/${ss_image}" "no"
check_file_exists "${input_dir}/${ss_region}" "no"
check_file_exists "${input_dir}/${baseline_image}" "no"
check_file_exists "${input_dir}/${baseline_region}" "no"

if [ "${MY_TEMP}" == "" ]
then
  temp_dir_prefix=~/temp/
else
  temp_dir_prefix=${MY_TEMP}
fi   

tmp_dir=`mktemp -d -q ${temp_dir_prefix}/__air_ss_all_timepoints.XXXXXX`
function cleanup
{
  echo "Cleaning up..."
  rm -rf  ${tmp_dir}
}
trap "cleanup" EXIT SIGINT SIGTERM SIGKILL 

# Register the baseline image to the ss_image. 
baseline_ss_input_file=${tmp_dir}/baseline_ss_input_file.txt
baseline_output_dir=${tmp_dir}/baseline
mkdir ${baseline_output_dir}
echo "${ss_image} ${ss_region} ${baseline_image} ${baseline_region}" > ${baseline_ss_input_file} 
regAIR-standard-space.sh ${input_dir} ${input_dir} ${baseline_ss_input_file} ${baseline_output_dir}
baseline_id=`echo ${baseline_image} | awk -F- '{printf $1}'`
ss_id=`echo ${ss_image} | awk -F- '{printf $1}'`
# Handle the -ve ID of the image by getting the image with the original ID and fixing the orientation. 
baseline_ss_image_wrong_orientation=${baseline_output_dir}/${baseline_id}-${ss_id}.img
baseline_ss_image=${output_dir}/${baseline_id}-075-1.img 
baseline_orient=`imginfo ${input_dir}/${baseline_image}.img -orient`
anchange ${baseline_ss_image_wrong_orientation} ${baseline_ss_image} -setorient ${baseline_orient} -study ${baseline_id} -series 75
# Handle the -ve ID of the region by searching in the reg-tmp output directory and setting the original ID. 
baseline_ss_region_large_id=`ls ${baseline_output_dir}/reg-tmp/???_*_*`
dims=`imginfo ${baseline_ss_image} -dims | awk '{printf "%d %d %d", $1, $2, $3}'`
regchange ${baseline_ss_region_large_id} ${output_dir}/. ${dims} -study ${baseline_id} -series 75 
baseline_ss_region=`ls ${output_dir}/???_${baseline_id}_*`
rm -rf ${baseline_output_dir}

# Perform all pairwise registration. 
for (( arg=${starting_arg}; arg<=$#; arg+=2 ))
do
  repeat_image=${!arg}
  (( arg_plus_1=arg+1 ))
  repeat_region=${!arg_plus_1}
  repeat_id=`echo ${repeat_image} | awk -F- '{printf $1}'`
  
  # Register the repeat image to the baseline_ss_image. 
  repeat_input_dir=${tmp_dir}/repeat_${arg}
  repeat_output_dir=${tmp_dir}/repeat_${arg}/results
  mkdir ${repeat_input_dir} ${repeat_output_dir}
  anchange ${baseline_ss_image} ${repeat_input_dir}/`basename ${baseline_ss_image}` -study ${baseline_id}
  cp ${baseline_ss_region} ${repeat_input_dir}/.
  anchange ${input_dir}/${repeat_image} ${repeat_input_dir}/`basename ${repeat_image}` -study ${repeat_id}
  cp ${input_dir}/${repeat_region} ${repeat_input_dir}/. 
  repeat_ss_input_file=${repeat_input_dir}/repeat_ss_input_file.txt
  echo "`basename ${baseline_ss_image%.img}` `basename ${baseline_ss_region}` `basename ${repeat_image}` `basename ${repeat_region}`" > ${repeat_ss_input_file}
  regAIR.sh ${repeat_input_dir} ${repeat_input_dir} ${repeat_ss_input_file} ${repeat_output_dir} -m 12 -d 8
  # Handle the -ve ID of the image by getting the image with the original ID and fixing the orientation. 
  repeat_ss_image_wrong_orientation=${repeat_output_dir}/${baseline_id}-${repeat_id}.img
  repeat_ss_image=${output_dir}/${repeat_id}-075-1.img 
  repeat_orient=`imginfo ${input_dir}/${repeat_image}.img -orient`
  anchange ${repeat_ss_image_wrong_orientation} ${repeat_ss_image} -setorient ${repeat_orient} -study ${repeat_id} -series 75
  # Handle the -ve ID of the region by searching in the reg-tmp output directory and setting the original ID. 
  repeat_ss_region_large_id=`ls ${repeat_output_dir}/reg-tmp/???_*_*`
  dims=`imginfo ${repeat_ss_image} -dims | awk '{printf "%d %d %d", $1, $2, $3}'`
  regchange ${repeat_ss_region_large_id} ${output_dir}/. ${dims} -study ${repeat_id} -series 75 
  repeat_ss_region=`ls ${output_dir}/???_${repeat_id}_*`
  rm -rf ${repeat_output_dir}
done

rm -rf ${tmp_dir}






