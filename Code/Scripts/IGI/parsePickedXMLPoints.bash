#! /bin/bash

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
	scans through the current directory for timestamped point pick files and 
	parses them to a single GSPointsLeft.txt and GSPointsRight.txt
EOF
exit 127
}

# Check args

check_for_help_arg "$*"
if [ $? -eq 1 ]; then
	Usage
fi

files=$(find * -maxdepth 0 -name "???????????????????_leftPoints.xml")

rm GSPointsLeft.txt

for file in $files
do
	frame=$( grep frame $file | cut -d \> -f 2 | cut -f 1 -d \<)
	points=$(grep -c "<point>" $file)
	point=1
	while [ $point -le $points ]
	do
		coord=$(grep -m ${point} -A 3 "<point>" ${file} | tail -n 1 | tr -d "[" | tr -d "]" | tr -d " " | tr -s "," "\t") 
		index=$(grep -m ${point} -A 1 "<point>" ${file} | tail -n 1 | cut -d \> -f 2 | cut -f 1 -d \<)
		echo -e ${frame}"\t"${index}"\t"${coord} | grep -v "\-1 \-1" >> GSPointsLeft.txt
		point=$(($point+1))
	done
done

files=$(find * -name "???????????????????_rightPoints.xml")

rm GSPointsRight.txt

for file in $files
do
	frame=$( grep frame $file | cut -d \> -f 2 | cut -f 1 -d \<)
	points=$(grep -c "<point>" $file)
	point=1
	while [ $point -le $points ]
	do
		coord=$(grep -m ${point} -A 3 "<point>" ${file} | tail -n 1 | tr -d "[" | tr -d "]" | tr -d " " | tr -s "," "\t")
		index=$(grep -m ${point} -A 1 "<point>" ${file} | tail -n 1 | cut -d \> -f 2 | cut -f 1 -d \<)
		echo -e ${frame}"\t"${index}"\t"${coord} | grep -v "\-1 \-1" >> GSPointsRight.txt
		point=$(($point+1))
	done
done


