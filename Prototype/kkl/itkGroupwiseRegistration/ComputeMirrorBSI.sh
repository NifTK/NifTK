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
#  Last Changed      : $LastChangedDate: 2010-08-24 16:48:21 +0100 (Tue, 24 Aug 2010) $ 
#  Revision          : $Revision: 3748 $
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

function Usage()
{
cat <<EOF

Script to compute mirror BSI.

EOF
exit 127
}

# Default params
ndefargs=5

# Check args
if [ $# -lt $ndefargs ]; then
  Usage
fi

set -x

# Get mandatory parameters
image=$1
brain_region=$2
hippo_l_region=$3
hippo_r_region=$4
output_dir=$5

mkdir -p $output_dir
execute_command "tmpdir=`mktemp -d -q ~/temp/mirror_bsi.XXXXXX`"
execute_command "mkdir -p ${tmpdir}/${output_dir}"
function cleanup
{
  echo "Cleaning up..."
  execute_command "rm -rf  ${tmpdir}"
}
trap "cleanup" EXIT SIGINT SIGTERM SIGKILL 

# check orientation. 
orientation=`imginfo ${image} -orient`
if [ "${orientation}" != "cor" ] 
then 
  echo "Sorry! Only handle coronal image for now."
  exit
fi 

id=`basename ${image} | awk -F- '{printf $1}'`

# flip the images and regions.
image_flip=${output_dir}/`basename ${image%.img}`_flip.img
brain_region_flip=${output_dir}/`basename ${brain_region}`_flip
hippo_l_region_flip=${output_dir}/`basename ${hippo_l_region}`_flip
hippo_r_region_flip=${output_dir}/`basename ${hippo_r_region}`_flip
anchange ${image} ${image_flip} -flipx 
dims=`imginfo ${image} -dims | awk '{printf "%s %s %s", $1, $2, $3}'`
regchange ${brain_region} ${brain_region_flip} ${dims} -flipx
regchange ${hippo_l_region} ${hippo_l_region_flip} ${dims} -flipx
regchange ${hippo_r_region} ${hippo_r_region_flip} ${dims} -flipx

# register them. 
echo "${image} ${brain_region} ${hippo_l_region} ${hippo_r_region} ${image_flip} ${brain_region_flip} ${hippo_l_region_flip} ${hippo_r_region_flip}" > ${output_dir}/input_${id}_reg.txt

export SKIP_SGE=1
ComputePairwiseRegistrationBatch.sh ${output_dir}/input_${id}_reg.txt ${output_dir} -symmetric sym_midway -dof 4 -similarity 4 -dilation 10 -pptol 0.001

# transform them. 
echo "${id}_l_groupwise 2 ${image} ${brain_region} ${hippo_l_region} dummy ${output}/${id}_pairwise_0_1_affine_second_roi1.dof ${image_flip} ${brain_region_flip} ${hippo_l_region_flip} dummy ${output}/${id}_pairwise_1_0_affine_second_roi1.dof" > ${output_dir}/input_${id}_transform_l.txt
echo "${id}_r_groupwise 2 ${image} ${brain_region} ${hippo_r_region} dummy ${output}/${id}_pairwise_0_1_affine_second_roi2.dof ${image_flip} ${brain_region_flip} ${hippo_r_region_flip} dummy ${output}/${id}_pairwise_1_0_affine_second_roi2.dof" > ${output_dir}/input_${id}_transform_r.txt

compute_symmetric_transform_batch.sh ${output_dir}/input_${id}_transform_l.txt ${output_dir} -tpn
compute_symmetric_transform_batch.sh ${output_dir}/input_${id}_transform_r.txt ${output_dir} -tpn

execute_command "rm ${tmpdir} -rf"










