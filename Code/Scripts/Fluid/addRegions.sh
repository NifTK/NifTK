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
#  Last Changed      : $LastChangedDate: 2011-10-19 18:24:09 +0100 (Wed, 19 Oct 2011) $ 
#  Revision          : $Revision: 7562 $
#  Last modified by  : $Author: kkl $
#
#  Original author   : m.clarkson@ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/

set -u
#set -x

source _niftkCommon.sh

# Note: The automatic doxygen generator uses the first two lines of the usage message.

function Usage()
{
cat <<EOF

Script to combine two midas regions together.

Usage: $0 target.img reg1.roi reg2.roi output.roi

Mandatory Arguments:

  target.img    : target image. 
  reg1.roi      : region 1
  reg2.roi      : region 2 
  output.roi    : output region (region1 + region2)
                       
EOF
exit 127
}

# Check args

check_for_help_arg "$*"
if [ $? -eq 1 ]; then
  Usage
fi

ndefargs=4

if [ $# -lt $ndefargs ]; then
  Usage
fi

target=$1
reg1=$2
reg2=$3
output=$4
option=$5

dilation="-d 3"

if [ "${option}" != "" ]
then 
  dilation=${option}
fi 

check_file_exists "${target}" no
check_file_exists "${reg1}" no
check_file_exists "${reg2}" no

execute_command "tmpdir=`mktemp -d -q /usr/tmp/add_region.XXXXXX`"

function cleanup
{
  echo "Cleaning up..."
  execute_command "rm -rf ${tmpdir}"
}
trap "cleanup" EXIT

execute_command "$MAKEMASK ${target} ${reg1} ${tmpdir}/reg1.img"
execute_command "$MAKEMASK ${target} ${reg2} ${tmpdir}/reg2.img ${dilation}"

execute_command "niftkInject -i ${tmpdir}/reg1.img -m ${tmpdir}/reg2.img -o ${tmpdir}/reg1.img"

execute_command "makeroi -img ${tmpdir}/reg1.img -out ${output} -alt 128"




 
 














