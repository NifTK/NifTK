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
#  Last Changed      : $LastChangedDate: 2011-12-15 13:50:36 +0000 (Thu, 15 Dec 2011) $ 
#  Revision          : $Revision: 8023 $
#  Last modified by  : $Author: kkl $
#
#  Original author   : leung@drc.ion.ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/
#
# Adapted from regAIR.sh.in. 
#
# Script to run our standard AIR linear registration to standard space,
# as specified in an input filename.
# 

#set -u 

function Usage()
{
cat <<EOF

This script is a convenient wrapper round our standard AIR to standard space.

Usage: $0 imageDir regionDir fileContainingImageNames outputDir [options ]

Mandatory Arguments:
 
  imageDir                 : is the directory containing your images
  regionDir                : is the directory containing your regions
  fileContainingImageNames : is a file containing:
  
                             baselineImage baselineMask repeatImage repeatMask
                             eg.
                             01727-003-1_resc        Mat_01727_1196694584    03061-003-1_resc        Pro_03061_1197570313

  outputDir                : is where the output is writen to.
  
Option:  
  -rreg_init               : Use rreg for initialisation. 

EOF
exit 127
}

# Check args
if [ $# -lt 4 ]; then
  Usage
fi


rreg_init="no"
if [ "$5" == "-rreg_init" ]
then 
  rreg_init="yes"
fi 

# Function   : Function will iterate through file fileContainingImageNames, 
#              and if $2="REGISTER" it will execute the registration.
# Param      : $1 filename, which should be fileContainingImageNames from command line
# Param      : $2 REGISTER if you want it to start registration, otherwise we just check the 
#              files exist.

function iterate_through_input_file()
{

  FILEOFDETAILS=$1
  DO_THE_REG=$2
  
  # Iterate through file. Each line should contain
  # baselineImg baselineMask repeatImage repeatMask

  n=`wc -l $FILEOFDETAILS | sed -n 's/^\(.*\) .*/\1/p'`
  i=0

  while [ "$i" -lt "$n" ] 
  do

    BASELINE_IMG=`awk 'NR-1 == '$i' {print $1}' $FILEOFDETAILS`
    BASELINE_MASK=`awk 'NR-1 == '$i' {print $2}' $FILEOFDETAILS`
    REPEAT_IMG=`awk 'NR-1 == '$i' {print $3}' $FILEOFDETAILS`
    REPEAT_MASK=`awk 'NR-1 == '$i' {print $4}' $FILEOFDETAILS`

    if [ "$DO_THE_REG" = "REGISTER" ]; then

      # Generate a file of all commands
      echo "_regAIR-standard-space.sh $REG_TMP_DIR $REG_RESULTS_DIR $IMAGE_DIR $BRAINREGION_DIR $BASELINE_IMG $BASELINE_MASK $REPEAT_IMG $REPEAT_MASK ${rreg_init}" >> regAIR_$$_commands.txt

    else
    
      echo "Checking the right files exist"
      
      echo "Checking for images $BASELINE_IMG, $REPEAT_IMG in $IMAGE_DIR and $BASELINE_MASK, $REPEAT_MASK in $BRAINREGION_DIR"
  
      check_file_exists $IMAGE_DIR/$BASELINE_IMG.img
      check_file_exists $IMAGE_DIR/$BASELINE_IMG.hdr
  
      check_file_exists $IMAGE_DIR/$REPEAT_IMG.img
      check_file_exists $IMAGE_DIR/$REPEAT_IMG.hdr
  
      check_file_exists $BRAINREGION_DIR/$BASELINE_MASK
      check_file_exists $BRAINREGION_DIR/$REPEAT_MASK

    fi
    
    # Increment loop counter
    i=$(($i+1))
  done
}

# Pick up command line parameters
IMAGE_DIR=$1
shift
BRAINREGION_DIR=$1
shift
FILEDETAILS=$1
shift
OUTPUT_DIR=$1
shift

# Setup the output directories

# This is just for temp images that can be deleted afterwards.
REG_TMP_DIR=$OUTPUT_DIR/reg-tmp

# And this is the main output dir, where all the results go.
REG_RESULTS_DIR=$OUTPUT_DIR

source _regInclude.sh

check_midas_env

check_directory_exists $OUTPUT_DIR

check_file_exists $FILEDETAILS

dos_2_unix $FILEDETAILS

echo "Running $0 with images from $IMAGE_DIR, brain regions from $BRAINREGION_DIR, image details from $FILEDETAILS, output=$OUTPUT_DIR"

# We first simply scan through file, cos then we can stop early if there are missing files
iterate_through_input_file $FILEDETAILS "CHECKONLY"

# We then iterate through file, generating commands to a file.
iterate_through_input_file $FILEDETAILS "REGISTER"

# Make output dirs
if [ ! -d $REG_TMP_DIR ]; then
  echo "Creating directory: $REG_TMP_DIR"
  mkdir $REG_TMP_DIR
fi

# Now run the file of commands.
run_batch_job regAIR_$$_commands.txt





