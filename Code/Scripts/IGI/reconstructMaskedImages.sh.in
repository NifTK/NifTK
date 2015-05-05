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

masks=$(ls *leftMask.png)

for mask in $masks 
do
	timestamp=${mask%_leftMask.png*}
	OK=1
	if [ -e ${timestamp}_rightMask.png ]
	then
		if [ -e ${timestamp}_left.png ]
		then
			if [ -e ${timestamp}_right.png ]
			then
				if [ -e ${timestamp}.match ]
				then
					OK=0
				else
					echo "Couldn't find ${timestamp}.match"
				fi
			else
				echo "Couldn't find ${timestamp}_right.png"
			fi
		else
			echo "Couldn't find ${timestamp}_left.png"
		fi
	else
		echo "Couldn't find ${timestamp}_rightMask.png"
	fi

	if [ $OK -eq 0 ]
	then
		echo Found everything I need for $mask
		niftkImageFeatureMatching --left ${timestamp}_left.png --right ${timestamp}_right.png --output ${timestamp}_pointPairs.txt

		niftkTriangulate2DPointPairsTo3D --inputPointPairs ${timestamp}_pointPairs.txt \
			--intrinsicRight ../scope_calibration/calib.right.intrinsic.txt \
			--intrinsicLeft ../scope_calibration/calib.left.intrinsic.txt \
			--rightToLeftExtrinsics ../scope_calibration/calib.r2l.txt \
			--outputPoints ${timestamp}_surface.mps \
			--leftMask ${timestamp}_leftMask.png \
			--rightMask ${timestamp}_rightMask.png \
			--trackerToWorld ${timestamp}.match \
			--leftLensToTracker ../scope_calibration/calib.left.handeye.txt
	fi
done

