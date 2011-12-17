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
#  Copyright (c) UCL : See the file LICENSE.txt in the top level 
#                      directory for futher details.
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/

inputFile=$1
directory=$2
outputDirectory=$3

if [ -z ${inputFile} ]; then
  echo "Anonymise images for DTI"
  echo "Usage: anonymise input_csv_file input_directory output_directory"
  echo "       where input_csv_file: csv file containing at least 3 columns, with study ID (2nd column) and subject code (3rd column)"
  echo "             input_directory: directory containing the input images"
  echo "             output_directory: directory storing the output images"
  exit
fi

cat ${inputFile} | while read eachLine 
do 
  studyID=`echo ${eachLine} | awk -F, '{printf $2}'`
  code=`echo ${eachLine} | awk -F, '{printf $3}'`
  if [ ! -z ${studyID} ]; then
    images=`find ${directory} -name *${studyID}-*.img`
    for eachImage in ${images}
    do
      anchange ${eachImage} ${outputDirectory}/`basename ${eachImage}` -name "${code}" -age 999 -sex m
    done
    
    bval=`find ${directory} -name *${studyID}-*.bval`
    bvec=`find ${directory} -name *${studyID}-*.bvec`
    dt_double=`find ${directory} -name *${studyID}-*.DT_Bdouble`
    scheme=`find ${directory} -name *${studyID}-*.scheme2`
    
    cp ${bval} ${bvec} ${dt_double} ${scheme} ${outputDirectory}
  fi
done
