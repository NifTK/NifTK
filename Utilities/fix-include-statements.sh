#!/bin/bash

#=============================================================================
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
#=============================================================================

# This script replaces the double quotes to angle brackets in the include
# statements if the included header is not in the same directory.

for sourceFile in `find . -name "*.c" ` `find . -name "*.h" ` `find . -name "*.cxx" ` `find . -name "*.hxx" ` `find . -name "*.txx" `
do
  egrep -q "#include *\"" $sourceFile
  if [ $? -eq 0 ]
  then
    egrep "#include *\"" $sourceFile > includes.txt
    while read includeLine
    do
      if [[ ${includeLine} =~ \#include.*\"(.*)\" ]]
      then
        headerFile=${BASH_REMATCH[1]}
        if [[ $headerFile =~ .*Exports.h ]] || [[ $headerFile =~ ui_.* ]]
        then
          continue
        fi
        sourceDir=${sourceFile%/*}
        headerFilePath=$sourceDir/$headerFile
        if [ ! -e $headerFilePath ]
        then
          headerFileWithEscapedSlashes=`echo "$headerFile" | sed 's/\//\\\\\//g'`
          sed -i "s/\#include.*\"$headerFileWithEscapedSlashes\".*/\#include <$headerFileWithEscapedSlashes>/" $sourceFile
          if [ $? -ne 0 ]
          then
            echo "Error occurred while executing the following command:"
            echo "sed command:" sed -i "s/\#include.*\"$headerFileWithEscapedSlashes\".*/\#include <$headerFileWithEscapedSlashes>/" $sourceFile
          fi
        fi
      fi
    done < includes.txt
  fi  
done

rm includes.txt
