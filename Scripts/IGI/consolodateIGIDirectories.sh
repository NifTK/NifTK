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

source _niftkCommon.sh

# Note: The automatic doxygen generator uses the first two lines of the usage message.

function Usage()
{
cat <<EOF
Create a single directory linking to all time stamped files in nameIn
Usage: consolodateIGIDirectories nameIn dirOut
       where nameIn is the name of the source to consolate eg Aurora_2/1
             dirOut is where to put the consolated result
EOF
exit 127
}

# Check args

check_for_help_arg "$*"
if [ $# -eq 0 ]; then
  Usage
fi

inputName=$1
outputDirectory=$2

mkdir -p $outputDirectory

directoriesIn=$(find * -type d -wholename "*$inputName")
for dir in $directoriesIn
do
	files=$(find $dir/* -name ???????????????????.*)
	here=$(pwd)
	cd $outputDirectory 
	mkdir -p $inputName
	cd $inputName
	for file in $files
	do
		ln -s ${here}/${file} .
	done
	cd ${here}
done

