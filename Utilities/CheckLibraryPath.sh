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
#  Last Changed      : $LastChangedDate: 2011-06-01 09:38:00 +0100 (Wed, 01 Jun 2011) $ 
#  Revision          : $Revision: 6322 $
#  Last modified by  : $Author: mjc $
#
#  Original author   : m.clarkson@ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/

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






