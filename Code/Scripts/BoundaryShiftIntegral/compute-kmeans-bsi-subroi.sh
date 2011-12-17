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
#  Last Changed      : $LastChangedDate: 2010-05-28 22:05:02 +0100 (Fri, 28 May 2010) $ 
#  Revision          : $Revision: 3326 $
#  Last modified by  : $Author: mjc $
#
#  Original author   : leung@drc.ion.ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/
#
# Script to run a NifTK K-Means BSI. 
# 

source _niftkCommon.sh

# Default params
ndefargs=7
baseline_image=
baseline_region=
repeat_image=
repeat_region=
output_dir=
use_dbc=0
kernel_size=5
no_norm=0

function Usage()
{
cat <<EOF

Script to compute BSI using automatic window selection.

Usage: $0 target.img target.roi source.img source.roi targetsub.roi sourcesub.roi output_dir [options]

Mandatory Arguments:

  target.img        : target (or baseline) image
  target.roi        : target (or baseline) region
  sourcesub.roi     : target (or baseline) subregion
  source.img        : source (or repeat) image
  source.roi        : source (or repeat) region
  sourcesub.roi     : source (or repeat) subregion
                    
  output_dir        : Output directory
  
                      1. *-kmeans-bsi.qnt file containing the BSI values.
                      2. *-seg.hdr file containing the k-means classifications.
                      
                      3. *-kmeans-bsi-dbc.qnt file containing the BSI after DBC correction (if selected). 
                      4. *-seg-dbc.hdr file containing the k-means classifications after DBC correction (if selected). 
                      
                    
Options:

  -dbc              : Do DBC.
  -kernel [int] [5] : Kernel size
  -no_norm          : Do not use intensity normalisation. 
                       
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
baseline_subroi=$5
repeat_subroi=$6
output_dir=$7

# Parse remaining command line options
shift $ndefargs
while [ "$#" -gt 0 ]
do
    case $1 in
    -dbc)
        use_dbc=1
	    ;;
    -no_norm)
        no_norm=1
      ;;
  	-kernel)
	    kernel_size=$2
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

if [ ! -f $baseline_subroi ]; then
    exitprog "Baseline subroi $baseline_subroi does not exist" 1
fi

if [ ! -f ${repeat_image}.img ]; then
    exitprog "Repeat image $repeat_image does not exist" 1
fi

if [ ! -f $repeat_region ]; then
    exitprog "Repeat region $repeat_region does not exist" 1
fi

if [ ! -f $repeat_subroi ]; then
    exitprog "Repeat subroi $repeat_subroi does not exist" 1
fi

if [ ! -d $output_dir ]; then
    exitprog "Output directory $output_dir does not exist" 1
fi

registered_dir_baseline=`dirname  $baseline_image`

execute_command "tmpdir=`mktemp -d -q /usr/tmp/kmeans-bsi.XXXXXX`"
execute_command "mkdir -p ${tmpdir}/${output_dir}"

baseline=`basename ${baseline_image} .img`
repeat=`basename ${repeat_image} .img`

execute_command "$COPY ${baseline_image}.img ${tmpdir}/baseline.img"
execute_command "$COPY ${baseline_image}.hdr ${tmpdir}/baseline.hdr"
execute_command "$COPY ${repeat_image}.img ${tmpdir}/repeat.img"
execute_command "$COPY ${repeat_image}.hdr ${tmpdir}/repeat.hdr"    
execute_command "$MAKEMASK ${tmpdir}/baseline.img ${baseline_region} ${tmpdir}/baseline_mask.img" 
execute_command "$MAKEMASK ${tmpdir}/repeat.img ${repeat_region} ${tmpdir}/repeat_mask.img"
execute_command "$MAKEMASK ${tmpdir}/baseline.img ${baseline_subroi} ${tmpdir}/baseline_subroi.img" 
execute_command "$MAKEMASK ${tmpdir}/repeat.img ${repeat_subroi} ${tmpdir}/repeat_subroi.img"

baseline_image_seg=${tmpdir}/${baseline}-seg
repeat_image_seg=${tmpdir}/${repeat}-seg
repeat_image_normalised=${output_dir}/${repeat}-normalised
log_file=${output_dir}/${baseline}-${repeat}-kmeans-bsi.qnt

if [ ${no_norm} -eq 1 ]
then
  execute_command "niftkKMeansWindowBSI \
    ${tmpdir}/baseline.hdr ${tmpdir}/baseline_mask.hdr \
    ${tmpdir}/repeat.hdr ${tmpdir}/repeat_mask.hdr \
    ${tmpdir}/baseline.hdr ${tmpdir}/baseline_subroi.hdr \
    ${tmpdir}/repeat.hdr ${tmpdir}/repeat_subroi.hdr \
    1 1 3 ${baseline_image_seg}.hdr ${repeat_image_seg}.hdr > ${log_file}"
else
  execute_command "niftkKMeansWindowWithLinearRegressionNormalisationBSI \
    ${tmpdir}/baseline.hdr ${tmpdir}/baseline_mask.hdr \
    ${tmpdir}/repeat.hdr ${tmpdir}/repeat_mask.hdr \
    ${tmpdir}/baseline.hdr ${tmpdir}/baseline_subroi.hdr \
    ${tmpdir}/repeat.hdr ${tmpdir}/repeat_subroi.hdr \
    1 1 3 -1 -1 ${baseline_image_seg}.hdr ${repeat_image_seg}.hdr ${repeat_image_normalised}.hdr > ${log_file}"
fi 

if [ $use_dbc -eq 1 ]; then

  baseline_image_dbc=${baseline}-dbc
  repeat_image_dbc=${repeat}-dbc
  baseline_image_seg_dbc=${tmpdir}/${baseline}-seg-dbc
  repeat_image_seg_dbc=${tmpdir}/${repeat}-seg-dbc
  repeat_image_normalised=${output_dir}/${repeat}-dbc-normalised
        
  log_file_dbc=${output_dir}/${baseline}-${repeat}-kmeans-bsi-dbc.qnt
  
  dilation=3
        
 execute_command "differentialbiascorrect ${tmpdir}/baseline ${tmpdir}/repeat ${baseline_region} ${repeat_region} ${output_dir} $kernel_size ${tmpdir} $baseline_image_dbc $repeat_image_dbc ${dilation} 0 0 0 0"
        
  if [ ${no_norm} -eq 1 ]
  then
    execute_command "niftkKMeansWindowBSI \
      ${output_dir}/${baseline_image_dbc}.hdr ${tmpdir}/baseline_mask.hdr \
      ${output_dir}/${repeat_image_dbc}.hdr ${tmpdir}/repeat_mask.hdr \
      ${output_dir}/${baseline_image_dbc}.hdr ${tmpdir}/baseline_subroi.hdr \
      ${output_dir}/${repeat_image_dbc}.hdr ${tmpdir}/repeat_subroi.hdr \
      1 1 3 ${baseline_image_seg_dbc}.hdr ${repeat_image_seg_dbc}.hdr > ${log_file_dbc}"
  else
    execute_command "niftkKMeansWindowWithLinearRegressionNormalisationBSI \
      ${output_dir}/${baseline_image_dbc}.hdr ${tmpdir}/baseline_mask.hdr \
      ${output_dir}/${repeat_image_dbc}.hdr ${tmpdir}/repeat_mask.hdr \
      ${output_dir}/${baseline_image_dbc}.hdr ${tmpdir}/baseline_subroi.hdr \
      ${output_dir}/${repeat_image_dbc}.hdr ${tmpdir}/repeat_subroi.hdr \
      1 1 3 -1 -1 ${baseline_image_seg_dbc}.hdr ${repeat_image_seg_dbc}.hdr ${repeat_image_normalised}.hdr > ${log_file_dbc}"
  fi 
        
fi

execute_command "rm ${tmpdir} -rf"
