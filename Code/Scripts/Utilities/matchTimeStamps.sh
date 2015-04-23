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
Scan time stamped files in 1st directory and find best match in 2nd directory
Outputs timing errors and creates symbolic links to matched files
Usage: matchTimeStamps.bash directory1 directory2
EOF
exit 127
}

# Check args

check_for_help_arg "$*"
if [ $? -eq 1 ]; then
  Usage
fi
dir1=$1
dir2=$2

if [ -z ${dir1} ]; then
  Usage
fi

if [ -z ${dir2} ]; then
  Usage
fi

timesIn=$(ls $dir1 | grep -c $)
matchTimes=$(ls $dir2 | grep -c $)

echo There are $timesIn files in dir 1 and $matchTimes in dir 2

timesIn=$(ls $dir1)
matchTimes=$(ls $dir2)

for time in $timesIn
do
	t1=${time%.*}
	echo best match for $t1 is ...
	bestDelta=$t1
	bestMatch=0
	for match in $matchTimes
	do
		t2=${match%.*}
		delta=$(echo "sqrt(($t1-$t2)^2)" | bc)
		if [ $delta -lt $bestDelta ]
		then
			bestMatch=$t2
			bestDelta=$delta
		fi
		if [ $delta -gt $bestDelta ]
		then
			break
		fi
	done
	echo $bestMatch with delta = $bestDelta
	ln -s $dir2/$bestMatch.txt $dir1/$t1.match
done

