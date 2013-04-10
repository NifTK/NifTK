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

#
# Script to run a NifTK K-Means BSI. 
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
use_dbc=0
kernel_size=5
no_norm=0
kn_dilation=3
calibration=0
niftkDBC=0
piecewise=0

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
                    
  output_dir        : Output directory
  
                      1. *-kmeans-bsi.qnt file containing the BSI values.
                      2. *-seg.hdr file containing the k-means classifications.
                      
                      3. *-kmeans-bsi-dbc.qnt file containing the BSI after DBC correction (if selected). 
                      4. *-seg-dbc.hdr file containing the k-means classifications after DBC correction (if selected). 
                      
                    
Options:

  -dbc              : Do DBC.
  -kernel [int] [5] : Kernel size
  -no_norm          : Do not use intensity normalisation. 
  -kn_dilation [3]  : Number of dilations for KN-BSI. 
  -calibration      : Perform self-calibration. 
  -niftkDBC         : Use niftkMTPDbc instead of the Midas DBC.
  -no_output_image  : Do not keep output images. 
                       
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
output_dir=$5

output_image_dir=${output_dir}

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
    -kn_dilation)
      kn_dilation=$2
      shift 1 
      ;;
    -calibration)
      calibration=1
      ;;
    -niftkDBC)
      niftkDBC=1
      ;;
    -no_output_image)
      output_image_dir="-no_output_image"
      ;;
    -piecewise)
      piecewise=1
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

execute_command "tmpdir=`mktemp -d -q /usr/tmp/kmeans-bsi.XXXXXX`"
execute_command "mkdir -p ${tmpdir}/${output_dir}"
function cleanup
{
  echo "Cleaning up..."
  execute_command "rm -rf  ${tmpdir}"
}
trap "cleanup" EXIT SIGINT SIGTERM SIGKILL 

if [ "${output_image_dir}" == "-no_output_image" ]
then 
  output_image_dir=${tmpdir}
fi 

baseline=`basename ${baseline_image} .img`
repeat=`basename ${repeat_image} .img`

execute_command "$COPY ${baseline_image}.img ${tmpdir}/baseline.img"
execute_command "$COPY ${baseline_image}.hdr ${tmpdir}/baseline.hdr"
execute_command "$COPY ${repeat_image}.img ${tmpdir}/repeat.img"
execute_command "$COPY ${repeat_image}.hdr ${tmpdir}/repeat.hdr"    
execute_command "$MAKEMASK ${tmpdir}/baseline.img ${baseline_region} ${tmpdir}/baseline_mask.img" 
execute_command "$MAKEMASK ${tmpdir}/repeat.img ${repeat_region} ${tmpdir}/repeat_mask.img"


baseline_image_seg=${tmpdir}/${baseline}-seg
repeat_image_seg=${tmpdir}/${repeat}-seg
repeat_image_normalised=${output_image_dir}/${repeat}-normalised
log_file=${output_dir}/${baseline}-${repeat}-kmeans-bsi.qnt

if [ ${no_norm} -eq 1 ]
then
  execute_command "niftkKMeansWindowBSI \
    ${tmpdir}/baseline.hdr ${tmpdir}/baseline_mask.hdr \
    ${tmpdir}/repeat.hdr ${tmpdir}/repeat_mask.hdr \
    ${tmpdir}/baseline.hdr ${tmpdir}/baseline_mask.hdr \
    ${tmpdir}/repeat.hdr ${tmpdir}/repeat_mask.hdr \
    1 1 ${kn_dilation} ${baseline_image_seg}.hdr ${repeat_image_seg}.hdr > ${log_file}"
else
  execute_command "niftkKMeansWindowWithLinearRegressionNormalisationBSI \
    ${baseline_image}.hdr ${tmpdir}/baseline_mask.hdr \
    ${repeat_image}.hdr ${tmpdir}/repeat_mask.hdr \
    ${baseline_image}.hdr ${tmpdir}/baseline_mask.hdr \
    ${repeat_image}.hdr ${tmpdir}/repeat_mask.hdr \
    1 1 ${kn_dilation} -1 -1 ${baseline_image_seg}.hdr ${repeat_image_seg}.hdr ${repeat_image_normalised}.hdr > ${log_file}"
fi 

if [ $use_dbc -eq 1 ]; then

  baseline_image_dbc=${baseline}_${repeat}-baseline-dbc
  repeat_image_dbc=${baseline}_${repeat}-repeat-dbc
  baseline_image_seg_dbc=${tmpdir}/${baseline}-seg-dbc
  repeat_image_seg_dbc=${tmpdir}/${repeat}-seg-dbc
  repeat_image_normalised=${output_image_dir}/${repeat}-dbc-normalised
        
  log_file_dbc=${output_dir}/${baseline}-${repeat}-kmeans-bsi-dbc.qnt
  log_file_dbc_robust=${output_dir}/${baseline}-${repeat}-kmeans-bsi-dbc-robust.qnt
  log_file_dbc_calibration=${output_dir}/${baseline}-${repeat}-kmeans-bsi-dbc-calibration.qnt
  log_file_dbc_reverse_calibration=${output_dir}/${baseline}-${repeat}-kmeans-bsi-dbc-reverse-calibration.qnt
  
  dilation=5
        
  if [ "${niftkDBC}" == "0" ] 
  then 
    execute_command "differentialbiascorrect ${tmpdir}/baseline ${tmpdir}/repeat ${baseline_region} ${repeat_region} ${output_image_dir} $kernel_size ${tmpdir} $baseline_image_dbc $repeat_image_dbc ${dilation} 0 0 0 0"
    
  else
    execute_command "$MAKEMASK ${tmpdir}/baseline.img ${baseline_region} ${tmpdir}/baseline_mask.img" 
    execute_command "$MAKEMASK ${tmpdir}/repeat.img ${repeat_region} ${tmpdir}/repeat_mask.img"
    execute_command "niftkMTPDbc -mode 2 -radius ${kernel_size} ${tmpdir}/baseline.hdr ${tmpdir}/baseline_mask.img ${output_image_dir}/${baseline_image_dbc}.hdr ${tmpdir}/repeat.hdr ${tmpdir}/repeat_mask.img ${output_image_dir}/${repeat_image_dbc}.hdr"
  fi 
        
  if [ ${no_norm} -eq 1 ]
  then
    execute_command "niftkKMeansWindowBSI \
      ${output_image_dir}/${baseline_image_dbc}.hdr ${tmpdir}/baseline_mask.hdr \
      ${output_image_dir}/${repeat_image_dbc}.hdr ${tmpdir}/repeat_mask.hdr \
      ${output_image_dir}/${baseline_image_dbc}.hdr ${tmpdir}/baseline_mask.hdr \
      ${output_image_dir}/${repeat_image_dbc}.hdr ${tmpdir}/repeat_mask.hdr \
      1 1 ${kn_dilation} ${baseline_image_seg_dbc}.hdr ${repeat_image_seg_dbc}.hdr > ${log_file_dbc}"
  else
    if [ "${piecewise}" == "0" ]
    then 
      execute_command "niftkKMeansWindowWithLinearRegressionNormalisationBSI \
        ${output_image_dir}/${baseline_image_dbc}.hdr ${tmpdir}/baseline_mask.hdr \
        ${output_image_dir}/${repeat_image_dbc}.hdr ${tmpdir}/repeat_mask.hdr \
        ${output_image_dir}/${baseline_image_dbc}.hdr ${tmpdir}/baseline_mask.hdr \
        ${output_image_dir}/${repeat_image_dbc}.hdr ${tmpdir}/repeat_mask.hdr \
        1 1 ${kn_dilation} -1 -1 ${baseline_image_seg_dbc}.hdr ${repeat_image_seg_dbc}.hdr ${repeat_image_normalised}.hdr > ${log_file_dbc}"
    else
      execute_command "niftkKMeansWindowWithPiecewiseNormalisationBSI \
        ${output_image_dir}/${baseline_image_dbc}.hdr ${tmpdir}/baseline_mask.hdr \
        ${output_image_dir}/${repeat_image_dbc}.hdr ${tmpdir}/repeat_mask.hdr \
        ${output_image_dir}/${baseline_image_dbc}.hdr ${tmpdir}/baseline_mask.hdr \
        ${output_image_dir}/${repeat_image_dbc}.hdr ${tmpdir}/repeat_mask.hdr \
        1 1 ${kn_dilation} -1 -1 ${baseline_image_seg_dbc}.hdr ${repeat_image_seg_dbc}.hdr ${repeat_image_normalised}.hdr > ${log_file_dbc}"
    fi 
  fi 
  
  if [ ${calibration} == 1 ]
  then
    calibration_scale=1.02
    scaling_factor=`echo "1.0/(${calibration_scale}*${calibration_scale}*${calibration_scale})-1.0" | bc -l`
    
    dims=`imginfo ${output_dir}/${baseline_image_dbc}.img -dims`
    dims_x=`echo ${dims} | awk '{printf $4}'`
    dims_y=`echo ${dims} | awk '{printf $5}'`
    dims_z=`echo ${dims} | awk '{printf $6}'`
    brain_com=`imginfo ${output_dir}/${baseline_image_dbc}.img -com -roi ${baseline_region}`
    brain_com_x=`echo ${brain_com} | awk '{printf $1}'`
    brain_com_y=`echo ${brain_com} | awk '{printf $2}'`
    brain_com_z=`echo ${brain_com} | awk '{printf $3}'`
    image_com=`imginfo ${output_dir}/${baseline_image_dbc}.img -com`
    image_com_x=`echo ${image_com} | awk '{printf $1}'`
    image_com_y=`echo ${image_com} | awk '{printf $2}'`
    image_com_z=`echo ${image_com} | awk '{printf $3}'`
    
    t_x=`echo "(${brain_com_x}-${image_com_x})*(-1.0+${calibration_scale})/${dims_x}" | bc -l`
    t_y=`echo "(${brain_com_y}-${image_com_y})*(-1.0+${calibration_scale})/${dims_y}" | bc -l`
    t_z=`echo "(${brain_com_z}-${image_com_z})*(-1.0+${calibration_scale})/${dims_z}" | bc -l`
    
    manualreslice << EOF
${t_x}
${t_y}
${t_z}
0
0
0
${calibration_scale}
${calibration_scale}
${calibration_scale}
n
${output_dir}/${baseline_image_dbc}.img
e
${output_dir}/${baseline_image_dbc}.img
y
${tmpdir}/calibration.air
y

EOF
    
    reslice ${tmpdir}/calibration.air ${output_dir}/calibration.img -o -k
if [ 1 == 1 ]
then     
    reslice ${tmpdir}/calibration.air ${output_dir}/calibration.img -o -n 10 -k
    regslice ${tmpdir}/calibration.air ${baseline_region} ${output_dir}/calibration_roi 3 -i 2
    makemask ${output_dir}/${baseline_image_dbc}.img ${output_dir}/calibration_roi ${tmpdir}/calibration_mask.img
    
    execute_command "niftkKMeansWindowWithLinearRegressionNormalisationBSI \
      ${output_dir}/${baseline_image_dbc}.hdr ${tmpdir}/baseline_mask.hdr \
      ${tmpdir}/calibration.img ${tmpdir}/calibration_mask.img \
      ${output_dir}/${baseline_image_dbc}.hdr ${tmpdir}/baseline_mask.hdr \
      ${tmpdir}/calibration.img ${tmpdir}/calibration_mask.img \
      1 1 ${kn_dilation} -1 -1 ${tmpdir}/bseg.hdr ${tmpdir}/rseg.hdr ${tmpdir}/rnorm.hdr > ${log_file_dbc_calibration}"
      
    echo "${scaling_factor}" >> ${log_file_dbc_calibration}
    
    makemask ${output_dir}/${baseline_image_dbc}.img ${baseline_region} ${output_dir}/baseline_bsi_image.img -k -bpp 16 -d 1
    makemask ${output_dir}/calibration.img ${output_dir}/calibration_roi ${output_dir}/repeat_bsi_image.img -k -bpp 16 -d 1
    
    execute_command "niftkKMeansWindowWithLinearRegressionNormalisationBSI \
      ${output_dir}/${baseline_image_dbc}.hdr ${tmpdir}/baseline_mask.hdr \
      ${output_dir}/calibration.img ${tmpdir}/calibration_mask.img \
      ${output_dir}/baseline_bsi_image.hdr ${tmpdir}/baseline_mask.hdr \
      ${output_dir}/repeat_bsi_image.hdr ${tmpdir}/calibration_mask.img \
      1 1 ${kn_dilation} -1 -1 ${tmpdir}/bseg.hdr ${tmpdir}/rseg.hdr ${tmpdir}/rnorm.hdr > ${log_file_dbc_calibration}"
    
    rm -f ${tmpdir}/calibration.hdr ${tmpdir}/calibration.img
    air_targets ${tmpdir}/calibration.air ${output_dir}/${repeat_image_dbc}.img ${output_dir}/${repeat_image_dbc}.img
    #reslice ${tmpdir}/calibration.air ${tmpdir}/calibration.img -o -n 10 -k
    #regslice ${tmpdir}/calibration.air ${repeat_region} ${tmpdir}/calibration_roi 3 -i 2
    makemask ${output_dir}/${repeat_image_dbc}.img ${tmpdir}/calibration_roi ${tmpdir}/calibration_mask.img
    
    execute_command "#niftkKMeansWindowWithLinearRegressionNormalisationBSI \
      ${output_dir}/${repeat_image_dbc}.img ${tmpdir}/repeat_mask.hdr \
      ${tmpdir}/calibration.img ${tmpdir}/calibration_mask.img \
      ${output_dir}/${repeat_image_dbc}.img ${tmpdir}/repeat_mask.hdr \
      ${tmpdir}/calibration.img ${tmpdir}/calibration_mask.img \
      1 1 ${kn_dilation} -1 -1 ${tmpdir}/bseg.hdr ${tmpdir}/rseg.hdr ${tmpdir}/rnorm.hdr > ${log_file_dbc_reverse_calibration}"
      
    echo "${scaling_factor}" >> ${log_file_dbc_reverse_calibration}
fi     
    
  fi 
  
        
fi

execute_command "rm ${tmpdir} -rf"




