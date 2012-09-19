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
#  Last Changed      : $LastChangedDate: 2010-05-28 22:05:02 +0100 (Fri, 28 May 2010) $ 
#  Revision          : $Revision: 3326 $
#  Last modified by  : $Author: mjc $
#
#  Original author   : m.clarkson@ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/

# Example script, to demonstrate how the Slicer Command Line Execution Model
# can be used to run any third part application within NiftyView. In this 
# example we run BET from FSL (http://www.fmrib.ox.ac.uk/fsl)

# The basic method is:
# If the user runs niftkBET.sh --xml we respond with the XML function contained herein.
# All other command line invocations, we pass the parameters onto the underlying program.

function XML()
{
cat <<EOF
<?xml version="1.0" encoding="utf-8"?>
<executable>
  <category>FSL.bet</category>
  <title>FSL (bet)</title>
  <description><![CDATA[A simple wrapper sript, provided within NifTK to enable FSL's 'bet' program to be run via the NiftyView GUI.]]></description>
  <version>0.0.1</version>
  <documentation-url>http://www.fmrib.ox.ac.uk/fsl</documentation-url>
  <license>BSD</license>
  <contributor>Matt Clarkson</contributor>
  <parameters advanced="false">
    <label>Images.</label>
    <description>Input and output images should be Nifti format.</description>
    <image fileExtensions="*.nii,*.nii.gz">      
      <name>inputImageName</name>
      <index>0</index>
      <description>Input image name</description>
      <label>Input image</label>
      <channel>input</channel>
    </image>    
    <image fileExtensions="*.nii,*.nii.gz">      
      <name>outputImageName</name>
      <index>1</index>
      <description>Output image name</description>
      <label>Output image</label>
      <channel>output</channel>
    </image>    
    <image fileExtensions="*.nii,*.nii.gz">      
      <name>outputBinaryMaskName</name>
      <flag>m</flag>
      <description>Output binary brain mask</description>
      <label>Output binary mask</label>
      <channel>output</channel>
    </image>    
  </parameters>
</executable>
EOF
exit 0
}

function Usage()
{
cat <<EOF
This script will run FSL's BET program.

Usage: 

EITHER:

  niftkBET.sh --xml
  
OR

  niftkBET.sh <any other parameters to pass to FSL's bet tool. See http://www.fmrib.ox.ac.uk/fsl>

EOF
exit 127
}

if [ $# -eq 0 ]; then
  Usage
fi

if [ $# -eq 1 -a "$1" = "--xml" ]; then
  XML
fi

which_bet=`which bet`
if [ "$which_bet" = "" ]; then
  echo "Could not find 'bet'. Please update your PATH, or speak to your systems administrator."
  exit 1
fi

command="bet $*"
echo "niftkBET.sh running command=:$command:"
eval ${command}

 