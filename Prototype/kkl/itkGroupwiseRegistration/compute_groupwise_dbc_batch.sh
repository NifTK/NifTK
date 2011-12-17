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
#  Last Changed      : $LastChangedDate: 2010-08-20 17:10:20 +0100 (Fri, 20 Aug 2010) $ 
#  Revision          : $Revision: 3732 $
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

# Default params
ndefargs=2
input_file=
output_dir=
use_dbc=0
kernel_size=5
dbc_mode=1

command_file="compute-mtp-dbc_"`date +"%Y%m%d-%H%M%S"`"_commands.txt"

script_dir=`dirname $0`

function Usage()
{
cat <<EOF

Wrapper to compute groupwise DBC, running in a batch.

Usage: $0 input_csv_file output_dir [options]

Mandatory Arguments:

  input_csv_file    : An input csv file containing the following format:
                      baseline_image,baseline_region,repeat_image,repeat_region
                    
                      For example:
                      02073-003-1_dbc,/var/lib/midas/data/adni-main/regions/brain/Liz_02073_1197552882,03913-003-1_dbc,/var/drc/scratch1/leung/testing/adni-3t-test/reg-tmp/Pro_93285003_1209670526
                    
                      Full paths should be specified. Images are assumed to be uncompressed.
                    
  output_dir        : Output directory
                    
Options:

  -kernel [int] [5] : Kernel size
  -mode [int] [1]   : Mode to calculate non-consecutive differential bias field. 
                        
EOF
exit 127 
}

function IterateThroughFile()
{
  local filename=$0
  local mode=$1
  local output=$2
  local kernel=$3  
  local dbc_mode=$4
  
  cat ${input_file} | while read each_line 
  do
    
    baseline_image=`echo ${each_line} | awk -F, '{printf $1}'`
    baseline_region=`echo ${each_line} | awk -F, '{printf $2}'`
    repeat_image=`echo ${each_line} | awk -F, '{printf $3}'`
    repeat_region=`echo ${each_line} | awk -F, '{printf $4}'`
    repeat_image2=`echo ${each_line} | awk -F, '{printf $5}'`
    repeat_region2=`echo ${each_line} | awk -F, '{printf $6}'`

    registered_dir_baseline=`dirname  $baseline_image`
    
    if [ "$mode" = "CHECK" ]; then

      check_file_exists ${baseline_image}.img
      check_file_exists ${baseline_region}
      check_file_exists ${repeat_image}.img
      check_file_exists ${repeat_region}
      check_file_exists ${repeat_image2}.img
      check_file_exists ${repeat_region2}
 
    else

      # Generate a file of all commands
      echo "${script_dir}/compute_groupwise_dbc.sh $baseline_image $baseline_region $repeat_image $repeat_region ${repeat_image2} ${repeat_region2} $output -kernel $kernel -mode ${dbc_mode}" >> $command_file 

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
    -kernel)
      kernel_size=$2
      shift 1
      ;;      
    -mode)
      dbc_mode=$2
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
IterateThroughFile $data_file "CALCULATE" $output_dir $kernel_size ${dbc_mode}

run_batch_job $command_file
