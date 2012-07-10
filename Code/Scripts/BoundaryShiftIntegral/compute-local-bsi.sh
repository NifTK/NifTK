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
#  Last Changed      : $LastChangedDate: 2011-12-14 15:33:49 +0000 (Wed, 14 Dec 2011) $ 
#  Revision          : $Revision: 8014 $
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
# Script to run a NifTK K-Means BSI. 
# 

#set -u 

source _niftkCommon.sh

# Default params
ndefargs=5
baseline_image=
baseline_region=
repeat_image=
repeat_region=
output_dir=
use_dw=0
cost_func=1
reg_dil=2
prealign_brain="yes"
just_bsi="no"
dbc="no"
use_kn="no"
use_sym="no"
min_window=-1
window_width_sd_factors="1 0.5"
brain_dof=6
local_dof=6

function Usage()
{
cat <<EOF

Script to local BSI using optional double intensity window. 

Usage: $0 baseline_image baseline_region repeat_image repeat_region baseline_local_region repeat_local_region output_dir
          output_study_id output_series_number output_echo_number [options]

Mandatory Arguments:

  baseline_image        : baseline image
  baseline_region       : baseline brain region
  repeat_image          : repeat image
  repeat_region         : repeat brain region
  baseline_local_region : baseline local region
  repeat_local_region   : repeat local region
  sub_roi               : sub region
  output_dir            : output directory
  output_study_id       : output study ID
  output_series_number  : output series ID
  output_echo_number    : output echo number
  weight_image          : input weight image
  
Options:

  -use_dw  : Use double intensity window. 
                       
  -cost_func x      : AIR local registration cost function 
                      1. standard deviation of ratio image (default)
                      2. least squares
                      3. least squares with intensity rescaling
                      
  -reg_dil x : number of dilations used for the registration (defalut: 2). 
  
  -no_prealign : Do not pre-align using brain mask. 
  
  -just_bsi : Just run BSI without registration. 
  
  -dbc : With DBC. 
  
  -min_window : Min. GM-WM window to be used only in double-window KN-BSI. 
  
  -window_width_sd_factors: Intensity window SD factor (default: "1 0.5"). 
                      
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
baseline_local_region=$5
repeat_local_region=$6
sub_roi=$7
output_dir=$8
output_study_id=$9
output_series_number=${10}
output_echo_number=${11}
weight_image=${12}


# Parse remaining command line options
shift $ndefargs
while [ "$#" -gt 0 ]
do
    case $1 in
    -use_dw)
        use_dw=1
      ;;
    -cost_func)
        shift
        cost_func=$1
      ;;
    -reg_dil)
        shift
        reg_dil=$1
      ;;
    -no_prealign)
        prealign_brain="no"
      ;;
    -just_bsi)
        just_bsi="yes"
      ;;
    -dbc)
        dbc="yes"
      ;;
    -use_kn)
        use_kn="yes"
      ;;
    -use_sym)
        use_sym="yes"
      ;;
    -min_window)
        shift
        min_window=$1
      ;;
    -window_width_sd_factors)
        shift 
        window_width_sd_factors=$1
      ;; 
    -dof) 
        shift
        brain_dof=$1
        shift
        local_dof=$1
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

if [ ! -f $baseline_local_region ] && [ "${baseline_local_region}" != "dummy" ]; then
    exitprog "Baseline local region $baseline_local_region does not exist" 1
fi

if [ ! -f $repeat_local_region ] && [ "${repeat_local_region}" != "dummy" ]; then
    exitprog "Repeat local region $repeat_local_region does not exist" 1
fi


if [ "${use_sym}" == "yes" ]
then 
  echo "Use symmetric registration..."

  execute_command "temp_dir=`mktemp -d -q /usr/tmp/__local-bsi.XXXXXX`"
  function cleanup
  {
    echo "Cleaning up..."
    execute_command "rm -rf ${temp_dir}"
  }
  trap "cleanup" EXIT
  
  baseline_brain_region_img=${temp_dir}/baseline_brain_region.img
  execute_command "${MAKEMASK} ${baseline_image}.img ${baseline_region} ${baseline_brain_region_img}"
  repeat_brain_region_img=${temp_dir}/repeat_brain_region.img
  execute_command "${MAKEMASK} ${repeat_image}.img ${repeat_region} ${repeat_brain_region_img}"
  baseline_local_region_img=${temp_dir}/baseline_local_region.img
  execute_command "${MAKEMASK} ${baseline_image}.img ${baseline_local_region} ${baseline_local_region_img}"
  repeat_local_region_img=${temp_dir}/repeat_local_region.img
  execute_command "${MAKEMASK} ${repeat_image}.img ${repeat_local_region} ${repeat_local_region_img}"
      
  dof_init_file="${output_dir}/100001-${output_study_id}-init.dof"
  dof_file="${output_dir}/100001-${output_study_id}.dof"
      
  niftkAffine \
      -ti ${baseline_local_region_img} -si ${repeat_local_region_img} \
      -tm ${baseline_local_region_img} \
      -sm ${repeat_local_region_img} \
      -ot ${dof_init_file} \
      -ri 2 -fi 3 -s 1 -tr 2 -o 6 -ln 3 -rmax 0.5 -rmin 0.1 -sym -d ${reg_dil} -stl 0 -spl 0
      
  niftkAffine \
      -ti ${baseline_image}.img -si ${repeat_image}.img \
      -tm ${baseline_local_region_img} \
      -sm ${repeat_local_region_img} \
      -it ${dof_init_file} \
      -ot ${dof_file} \
      -ri 2 -fi 3 -s 4 -tr 2 -o 6 -ln 1 -rmax 0.5 -rmin 0.1 -sym -d ${reg_dil}
      
  # Transform the images to mid-point. 
  itkComputeInitialAffineAtlas \
    ${output_dir}/${output_study_id}-%i-average-affine.hdr \
    ${baseline_image}.img \
    4 0 0 \
    ${repeat_image}.img ${dof_file}
    
  registered_baseline_image=${output_dir}/${output_study_id}-0-average-affine.hdr
  registered_repeat_image=${output_dir}/${output_study_id}-1-average-affine.hdr
  niftkAbsImageFilter -i ${registered_baseline_image} -o ${registered_baseline_image}
  niftkAbsImageFilter -i ${registered_repeat_image} -o ${registered_repeat_image}
  
  # Transform the mask to the affine "mid-point". 
  itkComputeInitialAffineAtlas \
    ${output_dir}/${output_study_id}-%i-average-affine-mask.hdr \
    ${baseline_brain_region_img} \
    2 0 0 \
    ${repeat_brain_region_img} ${dof_file}
  registered_baseline_mask=${output_dir}/${output_study_id}-0-average-affine-mask.hdr
  registered_repeat_mask=${output_dir}/${output_study_id}-1-average-affine-mask.hdr
  niftkAbsImageFilter -i ${registered_baseline_mask} -o ${registered_baseline_mask}
  niftkAbsImageFilter -i ${registered_repeat_mask} -o ${registered_repeat_mask}
  niftkThreshold -i ${registered_baseline_mask} -o ${registered_baseline_mask} -l 0 -u 128 -in 0 -out 255
  niftkThreshold -i ${registered_repeat_mask} -o ${registered_repeat_mask} -l 0 -u 128 -in 0 -out 255
  
  # Transform the local mask to the affine "mid-point". 
  itkComputeInitialAffineAtlas \
    ${output_dir}/${output_study_id}-%i-average-affine-local-mask.hdr \
    ${baseline_local_region_img} \
    2 0 0 \
    ${repeat_local_region_img} ${dof_file}
  registered_baseline_local_mask=${output_dir}/${output_study_id}-0-average-affine-local-mask.hdr
  registered_repeat_local_mask=${output_dir}/${output_study_id}-1-average-affine-local-mask.hdr
  niftkAbsImageFilter -i ${registered_baseline_local_mask} -o ${registered_baseline_local_mask}
  niftkAbsImageFilter -i ${registered_repeat_local_mask} -o ${registered_repeat_local_mask}
  niftkThreshold -i ${registered_baseline_local_mask} -o ${registered_baseline_local_mask} -l 0 -u 128 -in 0 -out 255
  niftkThreshold -i ${registered_repeat_local_mask} -o ${registered_repeat_local_mask} -l 0 -u 128 -in 0 -out 255
  
  baseline_image_basename=`basename ${baseline_image}`
  baseline_image_dbc=${baseline_image_basename}-dbc
  repeat_image_dbc=${output_study_id}-${output_series_number}-${output_echo_number}-dbc
  dilation=3
  kernel_size=5
  bsi_baseline_image=${output_dir}/${baseline_image_dbc}.img
  bsi_repeat_image=${output_dir}/${repeat_image_dbc}.img
      
  execute_command "niftkDbc -i1 ${registered_baseline_image} -m1 ${registered_baseline_mask} -i2 ${registered_repeat_image} -m2 ${registered_repeat_mask} -o1 ${output_dir}/${baseline_image_dbc}.hdr -o2 ${output_dir}/${repeat_image_dbc}.hdr -radius ${kernel_size}"
  
  baseline_image_seg=${temp_dir}/bseg.hdr
  repeat_image_seg=${temp_dir}/rseg.hdr
  xor_img=${temp_dir}/xor.hdr
  output_dw_qnt_file=${output_dir}/${output_study_id}-${dbc}DBC-${use_kn}kn-mw${min_window}-bsi-dw.qnt
  output_dw_global_intensity_qnt_file=${output_dir}/${output_study_id}-${dbc}DBC-${use_kn}kn-mw${min_window}-bsi-dw-global-mean.qnt

  execute_command "niftkKNDoubleWindowBSI \
      ${output_dir}/${baseline_image_dbc}.hdr ${registered_baseline_mask} \
      ${output_dir}/${repeat_image_dbc}.hdr ${registered_repeat_mask} \
      ${output_dir}/${baseline_image_dbc}.hdr ${registered_baseline_local_mask} \
      ${output_dir}/${repeat_image_dbc}.hdr ${registered_repeat_local_mask} \
      1 1 3   \
      ${baseline_image_seg} ${repeat_image_seg} \
      dummy ${xor_img} ${registered_baseline_local_mask} ${registered_repeat_local_mask} ${window_width_sd_factors} dummy ${min_window} > ${output_dw_qnt_file}"
  
  exit
fi   


if [ ${just_bsi} == "no" ]
then
  # Local registration and local classic BSI. 
  execute_command "reg-template-loc ${baseline_image}.img ${repeat_image}.img ${baseline_region} ${repeat_region} \
      ${baseline_local_region} ${repeat_local_region} \
      ${output_dir} ${output_dir} ${output_dir} ${output_dir} \
      ${output_study_id} ${output_series_number} ${output_echo_number} 0.45 0.65 \
      no ${output_study_id}_xor.roi ${reg_dil} 1 1 ${prealign_brain} ${brain_dof} ${local_dof} ${cost_func}"
else
  echo "Just running BSI"    
fi 
      
if [ ${use_dw} != 0 ]
then      
  execute_command "temp_dir=`mktemp -d -q /usr/tmp/__local-bsi.XXXXXX`"
  function cleanup
  {
    echo "Cleaning up..."
    execute_command "rm -rf ${temp_dir}"
  }
  trap "cleanup" EXIT
      
  baseline_brain_region_img=${temp_dir}/baseline_brain_region.img
  execute_command "${MAKEMASK} ${baseline_image}.img ${baseline_region} ${baseline_brain_region_img}"
  
  registered_repeat=${output_dir}/${output_study_id}-${output_series_number}-${output_echo_number}.img
  registered_repeat_region_img=${temp_dir}/registered_repeat_region.img
  if [ ! -f "${output_dir}/${output_study_id}_repeat_brain_region_registered" ]
  then 
    echo "Reslicing registered repeat brain region"
    execute_command "repeat_brain_region_basename=`basename ${repeat_region}`"
    execute_command "$COPY ${repeat_region} ${temp_dir}/${repeat_brain_region_basename}"
    registered_repeat_region=${temp_dir}/registered_repeat_region
    air_file="${output_dir}/100001-${output_study_id}.air"
    execute_command "regslice ${air_file} ${temp_dir}/${repeat_brain_region_basename} ${registered_repeat_region} ${output_series_number} -i 2"
  else
    echo "Using pre-sliced registered repeat brain region"
    registered_repeat_region="${output_dir}/${output_study_id}_repeat_brain_region_registered"
  fi 
  execute_command "${MAKEMASK} ${registered_repeat} ${registered_repeat_region} ${registered_repeat_region_img}"
  
  baseline_local_region_img=${temp_dir}/baseline_local_region.img
  execute_command "${MAKEMASK} ${baseline_image}.img ${baseline_local_region} ${baseline_local_region_img}"
  
  # By default, use the baseline local region as the repeat local region. 
  registered_repeat_local_region_img=${temp_dir}/baseline_local_region.img
  if [ ${repeat_local_region} != "dummy" ]
  then 
    registered_repeat_local_region_img=${temp_dir}/repeat_local_region.img
    registered_repeat_local_region=${temp_dir}/registered_repeat_local_region
    execute_command "regslice ${air_file} ${repeat_local_region} ${registered_repeat_local_region} ${output_series_number} -i 2"
    execute_command "${MAKEMASK} ${registered_repeat} ${registered_repeat_local_region} ${registered_repeat_local_region_img}"
  fi 
  
  sub_roi_img="dummy"
  sub_roi_name=""
  if [ "${sub_roi}" != "dummy" ]
  then
    sub_roi_img=${temp_dir}/sub_roi.img
    execute_command "${MAKEMASK} ${baseline_image}.img ${sub_roi} ${sub_roi_img}"
    sub_roi_name=`basename ${sub_roi}`
  fi 
  
  baseline_image_seg=${temp_dir}/bseg.hdr
  repeat_image_seg=${temp_dir}/rseg.hdr
  repeat_image_normalised=${temp_dir}/n.hdr
  #xor_img=${temp_dir}/xor.hdr
  # test output the xor.
  xor_img=${output_dir}/${output_study_id}-xor.hdr
  weight_image_name=`basename ${weight_image}`
  
  output_dw_qnt_file=${output_dir}/${output_study_id}-${sub_roi_name}-${weight_image_name}-${dbc}DBC-${use_kn}kn-mw${min_window}-bsi-dw.qnt
  output_classic_qnt_file=${output_dir}/${output_study_id}-${sub_roi_name}-${weight_image_name}-${dbc}DBC-${use_kn}kn-classic-bsi.qnt
  output_dw_global_intensity_qnt_file=${output_dir}/${output_study_id}-${dbc}DBC-${use_kn}kn-mw${min_window}-bsi-dw-global-mean.qnt
  
  bsi_baseline_image=${baseline_image}.img
  bsi_repeat_image=${registered_repeat}
  
  if [ "${dbc}" == "yes" ] 
  then 
    baseline_image_basename=`basename ${baseline_image}`
    baseline_image_dbc=${baseline_image_basename}-dbc
    repeat_image_dbc=${output_study_id}-${output_series_number}-${output_echo_number}-dbc
    dilation=3
    kernel_size=5
    bsi_baseline_image=${output_dir}/${baseline_image_dbc}.img
    bsi_repeat_image=${output_dir}/${repeat_image_dbc}.img
    
    if [ ! -f "${bsi_baseline_image}" ] || [ ! -f "${bsi_repeat_image}" ] 
    then 
      execute_command "differentialbiascorrect ${baseline_image}.img ${registered_repeat} ${baseline_region} ${registered_repeat_region} ${output_dir} ${kernel_size} ${temp_dir} ${baseline_image_dbc} ${repeat_image_dbc} ${dilation} 0 0 0 0"
    fi 
    
  fi   
  
  if [ "${use_kn}" == "yes" ] 
  then
    execute_command "niftkKMeansWindowWithLinearRegressionNormalisationBSI \
        ${bsi_baseline_image%.img}.hdr ${baseline_brain_region_img%.img}.hdr \
        ${bsi_repeat_image%.img}.hdr ${registered_repeat_region_img%.img}.hdr \
        ${bsi_baseline_image%.img}.hdr ${baseline_local_region_img%.img}.hdr \
        ${bsi_repeat_image%.img}.hdr ${registered_repeat_local_region_img%.img}.hdr \
        1 1 3 -1 -1 ${baseline_image_seg} ${repeat_image_seg} ${repeat_image_normalised} > ${output_classic_qnt_file}" 
        
    if [ ${repeat_local_region} != "dummy" ]
    then
      repeat_local_normalisation_region=${registered_repeat_local_region_img}
    else
      repeat_local_normalisation_region="dummy"
    fi 
    execute_command "niftkKNDoubleWindowBSI \
        ${bsi_baseline_image%.img}.hdr ${baseline_brain_region_img%.img}.hdr \
        ${bsi_repeat_image%.img}.hdr ${registered_repeat_region_img%.img}.hdr \
        ${bsi_baseline_image%.img}.hdr ${baseline_local_region_img%.img}.hdr \
        ${bsi_repeat_image%.img}.hdr ${registered_repeat_local_region_img%.img}.hdr \
        1 1 3   \
        ${baseline_image_seg} ${repeat_image_seg} \
        ${sub_roi_img} ${xor_img} ${baseline_local_region_img} ${repeat_local_normalisation_region} ${window_width_sd_factors} ${weight_image} ${min_window} > ${output_dw_qnt_file}"
        
    execute_command "niftkKNDoubleWindowBSI \
        ${bsi_baseline_image%.img}.hdr ${baseline_brain_region_img%.img}.hdr \
        ${bsi_repeat_image%.img}.hdr ${registered_repeat_region_img%.img}.hdr \
        ${bsi_baseline_image%.img}.hdr ${baseline_local_region_img%.img}.hdr \
        ${bsi_repeat_image%.img}.hdr ${registered_repeat_local_region_img%.img}.hdr \
        1 1 3   \
        ${baseline_image_seg} ${repeat_image_seg} \
        ${sub_roi_img} ${xor_img} dummy dummy ${window_width_sd_factors} ${weight_image} ${min_window} > ${output_dw_global_intensity_qnt_file}"
        
  else
    execute_command "niftkBSI \
        ${bsi_baseline_image%.img}.hdr ${baseline_brain_region_img%.img}.hdr \
        ${bsi_repeat_image%.img}.hdr ${registered_repeat_region_img%.img}.hdr \
        ${bsi_baseline_image%.img}.hdr ${baseline_local_region_img%.img}.hdr \
        ${bsi_repeat_image%.img}.hdr ${registered_repeat_local_region_img%.img}.hdr \
        1 1 0.45 0.65 > ${output_classic_qnt_file}" 
    
    execute_command "niftkDoubleWindowBSI \
        ${bsi_baseline_image%.img}.hdr ${baseline_brain_region_img%.img}.hdr \
        ${bsi_repeat_image%.img}.hdr ${registered_repeat_region_img%.img}.hdr \
        ${bsi_baseline_image%.img}.hdr ${baseline_local_region_img%.img}.hdr \
        ${bsi_repeat_image%.img}.hdr ${registered_repeat_local_region_img%.img}.hdr \
        1 1 3   \
        ${baseline_image_seg} ${repeat_image_seg} \
        -1 -1 -1 -1 \
        ${sub_roi_img} ${xor_img} ${baseline_local_region_img} ${window_width_sd_factors} ${weight_image} > ${output_dw_qnt_file}"
  fi         
      
  #makeroi -img ${output_dir}/${output_study_id}-xor.img -out ${output_dir}/${output_study_id}-xor -alt 0
  rm -f ${output_dir}/${output_study_id}-xor.img ${output_dir}/${output_study_id}-xor.hdr
  
fi 




















