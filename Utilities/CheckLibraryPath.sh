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

# Assumed input = PATH library
# eg. CheckLibraryPath blah:blah:blah libblah.so

DIRECTORIES=`echo $1 | tr ":" "\n"`
LIBRARY=$2

echo "Checking dependencies of library ${LIBRARY}"
echo " in paths:"
for f in ${DIRECTORIES}
do
  echo " ${f}"
done


DEPENDENCIES=`ldd ${LIBRARY} | grep "not found" | cut -f 1 -d "=" | sort -u`
for f in ${DEPENDENCIES}
do
  echo " depends on ${f}"
done

for f in ${DEPENDENCIES}
do
  found=0
  for g in ${DIRECTORIES}
  do
    if [ `ls ${g} | grep ${f} | wc -l ` -gt 0 ]; then
      echo "Found ${f} in ${g}"
      found=1
    fi
  done
  if [ ${found} -eq 0 ]; then
    echo "Didn't find ${f}"
  fi
done






