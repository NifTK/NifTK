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

Script to compute symmetric transform and BSI.

Usage: $0 target.img target.roi target.dof source.img source.roi source.dof output_dir [options]

Mandatory Arguments:

  target.img        : target (or baseline) image
  target.roi        : target (or baseline) region
  target.dof        : target (or baseline) dof
  source.img        : source (or repeat) image
  source.roi        : source (or repeat) region
  source.dof        : source (or repeat) dof

  output_dir        : Output directory
                       
EOF
exit 127
}

# Default params
ndefargs=8

# Check args
if [ $# -lt $ndefargs ]; then
  Usage
fi

set -x

# Get mandatory parameters

baseline_image=$1
baseline_region=$2
baseline_dof=$3
repeat_image=$4
repeat_region=$5
repeat_dof=$6
output_dir=$7
output_prefix=$8

asym_dof=""
interpolation=4

mkdir -p $output_dir
execute_command "tmpdir=`mktemp -d -q ~/temp/compute_symmetric_transform.XXXXXX`"
execute_command "mkdir -p ${tmpdir}/${output_dir}"
function cleanup
{
  echo "Cleaning up..."
  execute_command "rm -rf  ${tmpdir}"
}
trap "cleanup" EXIT SIGINT SIGTERM SIGKILL 

identity_dof=${tmpdir}/identity.dof

# Parse remaining command line options
shift $ndefargs
while [ "$#" -gt 0 ]
do
    case $1 in
    -asym)
      asym_dof="${identity_dof} 1 ${identity_dof} 1 "  
      ;;
    -interpolation)
      interpolation=$2
      shift 1
      ;;
    -*)
        Usage
      exitprog "Error: option $1 not recognised" 1
      ;;
    esac
    shift 1
done

if [ "${asym_dof}" != "" ] 
then 
  niftkCreateTransformation -type 3 -ot ${identity_dof}  0 0 0 0 0 0 1 1 1 0 0 0
fi 

# Get the inverse of repeat transform.
repeat_dof_inverse=${tmpdir}/repeat_inverse.dof
niftkInvertTransformation ${repeat_dof} ${repeat_dof_inverse}

# Compute the mean transform for baseline image.
average_transform=${output_dir}/${output_prefix}_average.dof
niftkComputeMeanTransformation ${average_transform} 1e-8 ${baseline_dof} 1 ${repeat_dof_inverse} 1 ${asym_dof}

# Do the transform.
resliced_image=${output_dir}/${output_prefix}_0.img
resliced_mask=${output_dir}/${output_prefix}_0_mask
baseline_region_img=${tmpdir}/baseline_mask.img
resliced_baseline_region_img=${tmpdir}/baseline_mask_resliced.img

makemask ${baseline_image} ${baseline_region} ${baseline_region_img}
niftkTransformation -ti ${baseline_image} -o ${resliced_image} -j ${interpolation} -g ${average_transform} -sym_midway ${baseline_image} ${repeat_image} -invertAffine
niftkAbsImageFilter -i ${resliced_image} -o ${resliced_image}
niftkTransformation -ti ${baseline_region_img} -o ${resliced_baseline_region_img} -j 2 -g ${average_transform} -sym_midway ${baseline_image} ${repeat_image} -invertAffine
makeroi -img ${resliced_baseline_region_img} -out ${resliced_mask} -alt 128

bsi_input_txt="${resliced_image%.img} ${resliced_mask}"

resliced_image=${output_dir}/${output_prefix}_1.img
resliced_mask=${output_dir}/${output_prefix}_1_mask
repeat_region_img=${tmpdir}/repeat_mask.img
resliced_repeat_region_img=${tmpdir}/repeat_mask_resliced.img

makemask ${repeat_image} ${repeat_region} ${repeat_region_img}
niftkTransformation -ti ${repeat_image} -o ${resliced_image} -j ${interpolation} -g ${average_transform} -sym_midway ${baseline_image} ${repeat_image}
niftkAbsImageFilter -i ${resliced_image} -o ${resliced_image}
niftkTransformation -ti ${repeat_region_img} -o ${resliced_repeat_region_img} -j 2 -g ${average_transform} -sym_midway ${baseline_image} ${repeat_image}
makeroi -img ${resliced_repeat_region_img} -out ${resliced_mask} -alt 128

bsi_input_txt="${bsi_input_txt} ${resliced_image%.img} ${resliced_mask}"

#echo ${bsi_input_txt} > ${tmpdir}/bsi_input.csv

execute_command "rm ${tmpdir} -rf"










