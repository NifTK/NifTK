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

# Default params
ndefargs=2
input_file=
output_dir=
use_dbc=0
kernel_size=5
command_file="compute-kmeans-bsi_$$_commands.txt"
kn_dilation=3
calibration=""
niftkDBC=""
no_output_image=""
piecewise=""


function Usage()
{
cat <<EOF

Wrapper to compute BSI using automatic window selection, running in a batch.

Usage: $0 input_csv_file output_dir [options]

Mandatory Arguments:

  input_csv_file    : An input csv file containing the following format:
                      baseline_image,baseline_region,repeat_image,repeat_region
                    
                      For example:
                      02073-003-1_dbc,/var/lib/midas/data/adni-main/regions/brain/Liz_02073_1197552882,03913-003-1_dbc,/var/drc/scratch1/leung/testing/adni-3t-test/reg-tmp/Pro_93285003_1209670526
                    
                      Full paths should be specified. Images are assumed to be uncompressed.
                    
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
  -calibration      : generate self-calibration result. 
  -niftkDBC         : Use niftkMTPDbc instead of the Midas DBC (must specify -dbc for it to work).
  -no_output_image  : Do not keep output images. 
  -piecewise        : Perferm piecewise regpression. 
                        
EOF
exit 127
}

function IterateThroughFile()
{
  local filename=$0
  local mode=$1
  local output=$2
  local kernel=$3  
  local usedbc=$4
  local no_norm=$5
  
  cat ${input_file} | while read each_line 
  do
    
    baseline_image=`echo ${each_line} | awk -F, '{printf $1}'`
    baseline_region=`echo ${each_line} | awk -F, '{printf $2}'`
    repeat_image=`echo ${each_line} | awk -F, '{printf $3}'`
    repeat_region=`echo ${each_line} | awk -F, '{printf $4}'`

    registered_dir_baseline=`dirname  $baseline_image`
    
    if [ "$mode" = "CHECK" ]; then

      check_file_exists ${baseline_image}.img
      check_file_exists ${baseline_region}
      check_file_exists ${repeat_image}.img
      check_file_exists ${repeat_region}
 
    else

      # Generate a file of all commands
      
      if [ $usedbc -eq 1 ]; then
        dbcArg="-dbc"
      fi
      
      echo "compute-kmeans-bsi.sh $baseline_image $baseline_region $repeat_image $repeat_region $output -kernel $kernel ${no_norm} -kn_dilation ${kn_dilation} $dbcArg ${calibration} ${niftkDBC} ${no_output_image} ${piecewise}" >> $command_file 

    fi

  done
}

# Check args

if [ $# -lt $ndefargs ]; then
  Usage
fi

# Get mandatory parameters

input_file=$1
output_dir=$2

# Parse remaining command line options
shift $ndefargs
while [ "$#" -gt 0 ]
do
    case $1 in
    -dbc)
        use_dbc=1
	    ;;
    -no_norm)
        no_norm="-no_norm"
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
      calibration="-calibration"
      ;;
    -niftkDBC)
      niftkDBC="-niftkDBC"
      ;;
    -no_output_image)
      no_output_image="-no_output_image"
      ;;
    -piecewise)
      piecewise="-piecewise"
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
IterateThroughFile $data_file "CHECK"

# Once to actually do it.
IterateThroughFile $data_file "CALCULATE" $output_dir $kernel_size $use_dbc ${no_norm} ${kn_dilation}

run_batch_job $command_file
