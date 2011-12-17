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
#
# Script to run a groupwise DBC. 
# 

source _niftkCommon.sh

set -x

# Default params
ndefargs=5
baseline_image=
baseline_region=
repeat_image=
repeat_region=
output_dir=
kernel_size=5
mode=1

function Usage()
{
cat <<EOF

Script to compute BSI using automatic window selection.

Usage: $0 target.img target.roi source.img source.roi output_dir [options]

Mandatory Arguments:

  target.img        : target (or baseline) image
  target.roi        : target (or baseline) region
  source.img        : source (or repeat) image
  source.roi        : source (or repeat) region
  source2.img       : source (or repeat) image
  source2.roi       : source (or repeat) region
                    
  output_dir        : Output directory
                    
Options:

  -kernel [int] [5] : Kernel size
  -mode [int] [1]   : Mode to calculate non-consecutive differential bias field. 
                       
EOF
exit 127
}

# Check args

if [ $# -lt $ndefargs ]; then
  Usage
fi

# Get mandatory parameters

baseline_image=$1
baseline_region=$2
repeat_image=$3
repeat_region=$4
repeat_image2=$5
repeat_region2=$6
output_dir=$7

# Parse remaining command line options
shift $ndefargs
while [ "$#" -gt 0 ]
do
    case $1 in
    -kernel)
      kernel_size=$2
      shift 1
      ;;      
    -mode)
      mode=$2
      shift 1
      ;;       
    -*)
        Usage
      exitprog "Error: option $1 not recognised" 1
      ;;
    esac
    shift 1
done

# Check command line arguments

if [ ! -f ${baseline_image}.img ]; then
    exitprog "Baseline image $baseline_image does not exist" 1
fi

if [ ! -f $baseline_region ]; then
    exitprog "Baseline region $baseline_region does not exist" 1
fi

if [ ! -f ${repeat_image}.img ]; then
    exitprog "Repeat image $repeat_image does not exist" 1
fi

if [ ! -f $repeat_region ]; then
    exitprog "Repeat region $repeat_region does not exist" 1
fi

if [ ! -d $output_dir ]; then
    exitprog "Output directory $output_dir does not exist" 1
fi

registered_dir_baseline=`dirname  $baseline_image`

execute_command "tmpdir=`mktemp -d -q /usr/tmp/groupwise_dbc.XXXXXX`"
execute_command "mkdir -p ${tmpdir}/${output_dir}"
function cleanup
{
  echo "Cleaning up..."
  execute_command "rm -rf  ${tmpdir}"
}
trap "cleanup" EXIT SIGINT SIGTERM SIGKILL 

baseline=`basename ${baseline_image} .img`
repeat=`basename ${repeat_image} .img`
repeat2=`basename ${repeat_image2} .img`

execute_command "$COPY ${baseline_image}.img ${tmpdir}/baseline.img"
execute_command "$COPY ${baseline_image}.hdr ${tmpdir}/baseline.hdr"
execute_command "$COPY ${repeat_image}.img ${tmpdir}/repeat.img"
execute_command "$COPY ${repeat_image}.hdr ${tmpdir}/repeat.hdr"    
execute_command "$COPY ${repeat_image2}.img ${tmpdir}/repeat2.img"
execute_command "$COPY ${repeat_image2}.hdr ${tmpdir}/repeat2.hdr"    
execute_command "$MAKEMASK ${tmpdir}/baseline.img ${baseline_region} ${tmpdir}/baseline_mask.img" 
execute_command "$MAKEMASK ${tmpdir}/repeat.img ${repeat_region} ${tmpdir}/repeat_mask.img"
execute_command "$MAKEMASK ${tmpdir}/repeat2.img ${repeat_region2} ${tmpdir}/repeat_mask2.img"

execute_command "niftkMTPDbc -mode ${mode} ${tmpdir}/baseline.hdr ${tmpdir}/baseline_mask.img ${output_dir}/${baseline}_dbc.hdr ${tmpdir}/repeat.hdr ${tmpdir}/repeat_mask.img ${output_dir}/${repeat}_dbc.hdr ${tmpdir}/repeat2.hdr ${tmpdir}/repeat_mask2.img ${output_dir}/${repeat2}_dbc.hdr"

execute_command "rm ${tmpdir} -rf"




