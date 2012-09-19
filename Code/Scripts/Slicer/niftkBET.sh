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
  <category>FSL.BET</category>
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
      <longflag>niftkBETInput</longflag>
      <description>Input image name</description>
      <label>Input image</label>
      <channel>input</channel>
    </image>    
    <image fileExtensions="*.nii,*.nii.gz">      
      <name>outputImageName</name>
      <longflag>niftkBETOutput</longflag>
      <description>Output image name</description>
      <label>Output image</label>
      <channel>output</channel>
    </image>
  </parameters>
  <parameters advanced="true">
    <label>Main bet2 options</label>
    <description>Specify any of the following</description>
    <image fileExtensions="*.nii,*.nii.gz">      
      <name>BrainSurfaceOutline</name>
      <flag>o</flag>
      <description>Generate brain surface outline overlaid onto original image</description>
      <label>Brain Surface Outline</label>
      <channel>output</channel>
    </image>              
    <image fileExtensions="*.nii,*.nii.gz">      
      <name>BinaryBrainMask</name>
      <flag>m</flag>
      <description>Generate binary brain mask</description>
      <label>Brain Binary Mask</label>
      <channel>output</channel>
    </image>
    <image fileExtensions="*.nii,*.nii.gz">      
      <name>ApproximateSkullImage</name>
      <flag>s</flag>
      <description>Generate approximate skull image</description>
      <label>Approximate Skull Image</label>
      <channel>output</channel>
    </image>     
    <boolean>
      <name>NoBrainImage</name>
      <flag>n</flag>      
      <description>Don't generate segmented brain image output</description>
      <label>No Brain Image</label>
    </boolean>
    <float>
      <name>FractionalIntensityThreshold</name>
      <flag>f</flag>
      <description>Fractional intensity threshold (0->1); default=0.5; smaller values give larger brain outline estimates</description>
      <label>Fractional intensity threshold</label>
      <default>0.5</default>
      <constraints>
        <minimum>0</minimum>
        <maximum>1</maximum>
        <step>0.1</step>
      </constraints>
    </float>
    <float>
      <name>GradientIntensityThreshold</name>
      <flag>g</flag>
      <description>Vertical gradient in fractional intensity threshold positive values give larger brain outline at bottom, smaller at top</description>
      <label>Fractional gradient</label>
      <default>0</default>
      <constraints>
        <minimum>-1</minimum>
        <maximum>1</maximum>
        <step>0.1</step>
      </constraints>
    </float>
    <point>
      <name>CentreGravity</name>
      <flag>c</flag>      
      <description>Centre-of-gravity (voxels not mm) of initial mesh surface.</description>
      <label>Centre of gravity</label>
    </point>
    <boolean>
      <name>Thresholding</name>
      <flag>t</flag>      
      <description>Apply thresholding to segmented brain image and mask</description>
      <label>Apply thresholding</label>
    </boolean>    
    <file fileExtensions="*.vtk,*.vtk">
      <name>GenerateMesh</name>
      <flag>e</flag>
      <description>Generates brain surface as mesh in .vtk format</description>
      <label>Generate mesh</label>
      <channel>output</channel>
    </file>
  </parameters>
  <parameters advanced="true">
    <label>Variations on default bet2 functionality (mutually exclusive options)</label>
    <description>mutually exclusive options</description>
    <boolean>
      <name>Robust</name>
      <flag>R</flag>      
      <description>Robust bran centre estimation (iterates BET several times)</description>
      <label>Robust</label>
    </boolean>
    <boolean>
      <name>OpticNerve</name>
      <flag>S</flag>      
      <description>Eye and optic nerve cleanup (can be useful in SIENA)</description>
      <label>Eye cleanup</label>
    </boolean>
    <boolean>
      <name>BiasField</name>
      <flag>B</flag>      
      <description>Bias field and neck cleanup (can be useful in SIENA)</description>
      <label>Bias field cleanup</label>
    </boolean>
    <boolean>
      <name>SmallZ</name>
      <flag>Z</flag>      
      <description>Improve BET if FOV is very small in Z (by temporarily padding end slices)</description>
      <label>Small Z</label>
    </boolean>
    <boolean>
      <name>FMRI</name>
      <flag>F</flag>      
      <description>Apply to 4D FMRI data (uses -f 0.3 and dilates brain mask slightly)</description>
      <label>4D FMRI</label>
    </boolean>
    <boolean>
      <name>AdditionalSkullScalp</name>
      <flag>A</flag>      
      <description>Run bet2 and then betsurf to get additional skull and scalp surfaces (includes registrations)</description>
      <label>Additional skull/scalp</label>
    </boolean>   
    <image fileExtensions="*.nii,*.nii.gz">      
      <name>AdditionalSkullScalpUsingT2</name>
      <longflag>A2</longflag>
      <description>as with -A, when also feeding in non-brain-extracted T2 (includes registrations)</description>
      <label>Additional T2 image</label>
      <channel>input</channel>
    </image>                                                                                                                                
  </parameters> 
  <parameters advanced="true">
    <label>Miscellaneous options</label>
    <description>Miscellaneous options</description>
    <boolean>
      <name>Verbose</name>
      <flag>v</flag>      
      <description>(switch on diagnostic messages)</description>
      <label>Verbose</label>
    </boolean>
    <boolean>
      <name>Debug</name>
      <flag>d</flag>      
      <description>(don't delete temporary intermediate images)</description>
      <label>Debug</label>
    </boolean>    
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

# Work around as the slicer XML does not allow you to specify parameter order,
# and coordinates are x,y,z whereas BET wants x y z.
INPUT_IMAGE=""
OUTPUT_IMAGE=""
OTHER_ARGS=""
while [ "$#" -gt 0 ]
do
  case $1 in
    --niftkBETInput)
        INPUT_IMAGE=$2
        shift 1
        ;;      
    --niftkBETOutput)
        OUTPUT_IMAGE=$2
        shift 1
        ;;
    -c)
        COORDINATES=`echo $2 | sed 's/\,/ /g'`
        OTHER_ARGS=" ${OTHER_ARGS} -c ${COORDINATES} "
        shift 1
        ;;
    *)
        OTHER_ARGS=" ${OTHER_ARGS} $1 "
        ;;    
  esac
  shift 1
done

command="bet $INPUT_IMAGE $OUTPUT_IMAGE $OTHER_ARGS"
echo "niftkBET.sh running command=:$command:"
eval ${command}

 