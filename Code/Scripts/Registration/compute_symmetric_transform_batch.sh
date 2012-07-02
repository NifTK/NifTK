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

#set -x

source _niftkCommon.sh

# Default params
ndefargs=2
input_file=
output_dir=
use_dbc=0
kernel_size=5
dbc_mode=1

command_file="compute_symmetric_transform_"`date +"%Y%m%d-%H%M%S"`.XXXXXXXXXX
mktemp ${command_file}

script_dir=`dirname $0`

function Usage()
{
cat <<EOF

Wrapper to compute symmetric transform and BSI, running in a batch.

Usage: $0 input_csv_file output_dir [options]

Mandatory Arguments:

  input_csv_file    : An input csv file containing the following format:
                      baseline_image,baseline_region,baseline_dof,repeat_image,repeat_region,repeat_dof
                    
                      Full paths should be specified. Images are assumed to be uncompressed.
                    
  output_dir        : Output directory
                    
Optional Arguments:                     
  -tp3                  : for 3 time points (old format). 
  -tpn                  : for n time points. 
  -interpolation i      : NN: 1, linear: 2, cubic: 3, sinc: 4. 
  -double_window yes/no : double window or not. 
  -ss_atlas dummy       : atlas used to generate img from midas region. 
  -just_dbc yes/no      : just do dbc without transformation if yes. 
                        
EOF
exit 127 
}

function IterateThroughFile()
{
  local filename=$0
  local mode=$1
  local output=$2
  
  cat ${input_file} | while read each_line 
  do
    if [ "${tpn}" != "" ] 
    then 
      if [ "$mode" != "CHECK" ]; then
        # Generate a file of all commands
        echo "${script_dir}/compute_symmetric_transform_n.sh ${asym_flag} ${interpolation} ${double_window} ${just_dbc} ${ss_atlas} ${output} ${each_line}" >> $command_file
      fi
    elif [ "${tp3}" == "" ] 
    then 
      baseline_image=`echo ${each_line} | awk -F, '{printf $1}'`
      baseline_region=`echo ${each_line} | awk -F, '{printf $2}'`
      baseline_dof=`echo ${each_line} | awk -F, '{printf $3}'`
      repeat_image=`echo ${each_line} | awk -F, '{printf $4}'`
      repeat_region=`echo ${each_line} | awk -F, '{printf $5}'`
      repeat_dof=`echo ${each_line} | awk -F, '{printf $6}'`
      output_prefix=`echo ${each_line} | awk -F, '{printf $7}'`
      if [ "$mode" = "CHECK" ]; then
        check_file_exists ${baseline_image}
        check_file_exists ${baseline_region}
        check_file_exists ${baseline_dof}
        check_file_exists ${repeat_image}
        check_file_exists ${repeat_region}
        check_file_exists ${repeat_dof}
      else
        # Generate a file of all commands
        echo "${script_dir}/compute_symmetric_transform.sh ${baseline_image} ${baseline_region} ${baseline_dof} ${repeat_image} ${repeat_region} ${repeat_dof} ${output} ${output_prefix} ${asym_flag} ${interpolation}" >> $command_file
      fi
    else
      image1=`echo ${each_line} | awk -F, '{printf $1}'`
      region1=`echo ${each_line} | awk -F, '{printf $2}'`
      dof1_2=`echo ${each_line} | awk -F, '{printf $3}'`
      dof1_3=`echo ${each_line} | awk -F, '{printf $4}'`
      image2=`echo ${each_line} | awk -F, '{printf $5}'`
      region2=`echo ${each_line} | awk -F, '{printf $6}'`
      dof2_1=`echo ${each_line} | awk -F, '{printf $7}'`
      dof2_3=`echo ${each_line} | awk -F, '{printf $8}'`
      image3=`echo ${each_line} | awk -F, '{printf $9}'`
      region3=`echo ${each_line} | awk -F, '{printf $10}'`
      dof3_1=`echo ${each_line} | awk -F, '{printf $11}'`
      dof3_2=`echo ${each_line} | awk -F, '{printf $12}'`
      output_prefix=`echo ${each_line} | awk -F, '{printf $13}'`
      if [ "$mode" = "CHECK" ]; then
        check_file_exists ${image1}
        check_file_exists ${region1}
        check_file_exists ${dof1_2}
        check_file_exists ${dof1_3}
        check_file_exists ${image2}
        check_file_exists ${region2}
        check_file_exists ${dof2_1}
        check_file_exists ${dof2_3}
        check_file_exists ${image3}
        check_file_exists ${region3}
        check_file_exists ${dof3_1}
        check_file_exists ${dof3_2}
      else
        # Generate a file of all commands
        echo "${script_dir}/compute_symmetric_transform_3.sh ${image1} ${region1} ${dof1_2} ${dof1_3} ${image2} ${region2} ${dof2_1} ${dof2_3} ${image3} ${region3} ${dof3_1} ${dof3_2} ${output} ${output_prefix} ${asym_flag} ${interpolation} ${pairwise}" >> $command_file
      fi
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

asym_flag=""
tp3=""
tpn=""
interpolation="-interpolation 4"
double_window="-double_window yes"
ss_atlas="-ss_atlas dummy"
pairwise="no"
just_dbc="-just_dbc no"

# Parse remaining command line options
shift $ndefargs
while [ "$#" -gt 0 ]
do
    case $1 in
    -asym)
      asym_flag="-asym"
      ;;
    -tp3)
      tp3="-tp3"
      ;;
    -tpn)
      tpn="-tpn"
      ;;
    -interpolation)
      interpolation="-interpolation $2"
      shift 1
      ;;
    -double_window)
      double_window="-double_window $2"
      shift 1
      ;;
    -ss_atlas)
      ss_atlas="-ss_atlas $2"
      shift 1
      ;;
    -pairwise)
      pairwise="-pairwise $2"
      shift 1
      ;; 
    -just_dbc)
      just_dbc="-just_dbc $2"
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

dos_2_unix ${input_file}

# Once to check all files exist
IterateThroughFile $data_file "CHECK" ${output_dir}

# Once to actually do it.
IterateThroughFile $data_file "CALCULATE" $output_dir

run_batch_job $command_file
