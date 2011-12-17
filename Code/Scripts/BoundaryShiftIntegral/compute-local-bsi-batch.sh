#!/bin/bash

#set -x

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
#  Last Changed      : $LastChangedDate: 2011-10-12 17:19:08 +0100 (Wed, 12 Oct 2011) $ 
#  Revision          : $Revision: 7503 $
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
# Script to run a local BSI. 
# 

# set -u

source _niftkCommon.sh

# Default params
ndefargs=3
input_file=
output_dir=
use_dw=""
cost_func="-cost_func 1"
command_file="compute-local-bsi_$$_commands.txt"
reg_dil="-reg_dil 2"
prealign_brain=""
just_bsi=""
dbc=""
use_kn=""
use_sym=""
min_window="-min_window -1"

function Usage()
{
cat <<EOF

Wrapper to compute local BSI using optional double intensity window, running in a batch.

Usage: $0 input_csv_file output_series_number output_dir [options]

Mandatory Arguments:

  input_csv_file    : An input csv file containing the following format:
                      baseline_image,baseline_brain_region,repeat_image,repeat_brain_region,baseline_local_region,repeat_local_region,sub_roi
                    
                      For example:
                      02073-003-1_dbc,/var/lib/midas/data/adni-main/regions/brain/Liz_02073_1197552882,03913-003-1_dbc,/var/drc/scratch1/leung/testing/adni-3t-test/reg-tmp/Pro_93285003_1209670526,baseline_hippo,repeat_hippo,sub_roi,weight_image
                    
                      Full paths should be specified. Images are assumed to be uncompressed.
                      You need to specify sub_roi and weight_image as dummy if no sub-ROI and weight_image are used. 
                      
  output_series_number : output series number
                    
  output_dir        : Output directory
  
                      1. *.qnt file containing the BSI values.
                    
Options:

  -use_dw           : Use double intensity window. 
  -cost_func x      : AIR local registration cost function 
                      1. standard deviation of ratio image (default)
                      2. least squares
                      3. least squares with intensity rescaling
  -reg_dil x : number of dilations used for the registration (defalut: 2). 
  
  -no_prealign : Do not pre-align using brain mask. 
  
  -just_bsi : Just run the BSI. 
  
  -use_kn : use k-means normalisation. 
  
  -use_sym : use NifTK symmetric registration and transformation. 
  
  -min_window : Min. GM-WM window to be used only in double-window KN-BSI. 
                      
EOF
exit 127
}


function IterateThroughFile()
{
  local input_file=$1
  local mode=$2
  local output=$3
  local output_series_number=$4
  local use_dw=$5
  local cost_func=$6
  local reg_dil=$7
  local prealign_brain=$8
  local just_bsi=$9
  local dbc=${10}
  local use_kn=${11}
  local use_sym=${12}
  local min_window=${13}
  
  cat ${input_file} | while read each_line 
  do
    
    local baseline_image=`echo ${each_line} | awk -F, '{printf $1}'`
    local baseline_region=`echo ${each_line} | awk -F, '{printf $2}'`
    local repeat_image=`echo ${each_line} | awk -F, '{printf $3}'`
    local repeat_region=`echo ${each_line} | awk -F, '{printf $4}'`
    local baseline_local_region=`echo ${each_line} | awk -F, '{printf $5}'`
    local repeat_local_region=`echo ${each_line} | awk -F, '{printf $6}'`
    local sub_roi=`echo ${each_line} | awk -F, '{printf $7}'`
    local weight_file=`echo ${each_line} | awk -F, '{printf $8}'`
    
    repeat_image_copy=`pwd`/`basename ${repeat_image}`

    if [ "$mode" = "CHECK" ]; then
      if [ ! -f ${repeat_image_copy}.img ]
      then 
        if [ -f ${repeat_image}.img ]
        then 
          anchange ${repeat_image}.img ${repeat_image_copy}.img -sex m
        fi 
        if [ -f ${repeat_image}.img.gz ]
        then 
          anchange ${repeat_image}.img.gz ${repeat_image_copy}.img -sex m
        fi
        if [ -f ${repeat_image}.img.gz ]
        then 
          anchange ${repeat_image}.img.Z ${repeat_image_copy}.img -sex m
        fi 
      fi 

      check_file_exists ${baseline_image}.img "no"
      check_file_exists ${baseline_region} "no"
      check_file_exists ${repeat_image_copy}.img "no"
      check_file_exists ${repeat_region} "no"
      check_file_exists ${baseline_local_region} "no"
      
    else
    
      baseline_basename=`basename ${baseline_image}`
      output_study_id=`echo ${baseline_basename} | awk -F- '{printf $1}'`
      output_echo_number=1
      # Generate a file of all commands
      echo "compute-local-bsi.sh $baseline_image $baseline_region ${repeat_image_copy} $repeat_region \
                                 ${baseline_local_region} ${repeat_local_region} \"${sub_roi}\" \
                                 ${output} ${output_study_id} ${output_series_number} ${output_echo_number} ${weight_file} \
                                 ${use_dw} ${cost_func} ${reg_dil} ${prealign_brain} ${just_bsi} ${dbc} ${use_kn} ${use_sym} ${min_window}" >> $command_file 
    fi

  done
}

# Check args

if [ $# -lt $ndefargs ]; then
  Usage
fi

# Get mandatory parameters

input_file=$1
output_series_number=$2
output_dir=$3

# Parse remaining command line options
shift $ndefargs
while [ "$#" -gt 0 ]
do
    case $1 in
    -use_dw)
        use_dw="-use_dw"
      ;;
    -cost_func)
        shift
        cost_func="-cost_func $1"
      ;;
    -reg_dil)
        shift
        reg_dil="-reg_dil $1"
      ;;
    -no_prealign)
        prealign_brain="-no_prealign"
      ;;
    -just_bsi)
        just_bsi="-just_bsi"
      ;;
    -dbc)
        dbc=$1
      ;;
    -use_kn)
        use_kn=$1
      ;;
    -use_sym) 
        use_sym=$1
      ;;
    -min_window)
        shift
        min_window="-min_window $1"
      ;;
    -*)
      Usage
      exitprog "Error: option $1 not recognised" 1
      ;;
    esac
    shift 1
done


# Check command line arguments
if [ ! -f $input_file ]; then
    exitprog "Input file $input_file does not exist" 1
fi

if [ ! -d $output_dir ]; then
    exitprog "Output directory $output_dir does not exist" 1
fi

dos_2_unix ${input_file}

# Once to check all files exist
IterateThroughFile ${input_file} "CHECK" ${output_dir} ${output_series_number} "${use_dw}" "${cost_func}" "${reg_dil}" "${prealign_brain}" "${just_bsi}" "${dbc}" "${use_kn}" "${use_sym}" "${min_window}"

# Once to actually do it.
IterateThroughFile ${input_file} "CALCULATE" ${output_dir} ${output_series_number} "${use_dw}" "${cost_func}" "${reg_dil}" "${prealign_brain}" "${just_bsi}" "${dbc}" "${use_kn}" "${use_sym}" "${min_window}"

run_batch_job $command_file




























