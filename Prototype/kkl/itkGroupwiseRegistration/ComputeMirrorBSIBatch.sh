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

source _niftkCommon.sh

# Default params
ndefargs=2
input_file=
output_dir=

command_file="ComputeMirrorBsi_"`date +"%Y%m%d-%H%M%S"`"_commands.txt"

script_dir=`dirname $0`

function Usage()
{
cat <<EOF

Wrapper to compute mirror BSI on the hippocampus, running in a batch.

Usage: $0 input_file output_dir [options]

Mandatory Arguments:

  input_file    : An input csv file containing the following format:
                      image,brain_region,left_hippo,right_hippo
                    
  output_dir    : Output directory
                        
EOF
exit 127 
}

function IterateThroughFile()
{
  local mode=$1
  local output=$2
  
  cat ${input_file} | while read each_line 
  do
    
    image=`echo ${each_line} | awk -F, '{printf $1}'`
    brain_region=`echo ${each_line} | awk -F, '{printf $2}'`
    left_hippo=`echo ${each_line} | awk -F, '{printf $3}'`
    right_hippo=`echo ${each_line} | awk -F, '{printf $4}'`
    
    if [ "$mode" = "CHECK" ]; then

      check_file_exists ${image}
      check_file_exists ${brain_region}
      check_file_exists ${left_hippo}
      check_file_exists ${right_hippo}
 
    else

      # Generate a file of all commands
      echo "${script_dir}/ComputeMirrorBSI.sh ${image} ${brain_region} ${left_hippo} ${right_hippo} ${output} " >> $command_file 

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

# Check command line arguments
if [ ! -f $input_file ]; then
    exitprog "Input file $input_file does not exist" 1
fi

if [ ! -d $output_dir ]; then
    exitprog "Output directory $output_dir does not exist" 1
fi

dos_2_unix ${input_file}

# Once to check all files exist
IterateThroughFile "CHECK"

# Once to actually do it.
IterateThroughFile "CALCULATE" $output_dir 

run_batch_job $command_file



