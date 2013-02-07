#!/bin/sh

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

# This script is to manually run the commands to package an installation directory
# into a standalone application bundle that can be copied to another Mac.
# Based on: http://doc.qt.nokia.com/qq/qq09-mac-deployment.html
# and Ludvig A Norin's comments on http://stackoverflow.com/questions/96882/how-do-i-create-a-nice-looking-dmg-for-mac-os-x-using-command-line-tools

if [ $# -ne 1 ]; then
  echo "Usage: MakeMacDMG.sh installFolder"
  exit
fi

# Create some working variables, used throughout script
INSTALL_DIR=$1
BASE_NAME=`basename $1`
DIR_NAME=`dirname $1`
OUTPUT_DMG=${BASE_NAME}.dmg
TMP_OUTPUT_DMG=tmp.${OUTPUT_DMG}
EFFECTIVE_ROOT=${DIR_NAME}/tmp
CWD=`pwd`

# Print some of these out, for debugging purposes
echo "Packing Mac application in ${INSTALL_DIR} into package named ${OUTPUT_DMG}"
echo "BASE_NAME=${BASE_NAME}"
echo "DIR_NAME=${DIR_NAME}"
echo "EFFECTIVE_ROOT=${EFFECTIVE_ROOT}"

# Cleanup previous stuff
if [ -d ${EFFECTIVE_ROOT} ] ; then
  \rm -rf ${EFFECTIVE_ROOT}
fi
if [ -f ${DIR_NAME}/${OUTPUT_DMG} ] ; then
  \rm -f ${DIR_NAME}/${OUTPUT_DMG}
fi
if [ -f ${DIR_NAME}/${TMP_OUTPUT_DMG} ] ; then
  \rm -f ${DIR_NAME}/${TMP_OUTPUT_DMG}
fi
 
# Copy the whole lot into temp folder, as this folder becomes the root in the final dmg package.
mkdir ${EFFECTIVE_ROOT}
cd ${DIR_NAME}
cp -r ${BASE_NAME} ${EFFECTIVE_ROOT}

# Credit to Ludvig A Norin for the post on this forum message
# http://stackoverflow.com/questions/96882/how-do-i-create-a-nice-looking-dmg-for-mac-os-x-using-command-line-tools

# Calculate required size of disk image
echo "Calculating required size"
initial_size=`du -ks ${EFFECTIVE_ROOT} | awk '{print $1}'`
SIZE=$(($initial_size+2000))
echo "Size of directory to mount=${SIZE}k"

# Create the disk image
echo "Creating disk image"
hdiutil create -srcfolder "${EFFECTIVE_ROOT}" -volname "${BASE_NAME}" -fs HFS+ -fsargs "-c c=64,a=16,e=16" -format UDRW -size ${SIZE}k ${TMP_OUTPUT_DMG}
      
# Mount the disk image
echo "Mounting disk image"
device=$(hdiutil attach -readwrite -noverify -noautoopen "${TMP_OUTPUT_DMG}" | egrep '^/dev/' | sed 1q | awk '{print $1}')

# Sleep for a bit to make sure mount works
sleep 4

# Optional: Add a background picture
echo "Setting background picture into image"
cd ${CWD}
backgroundPictureName=ucl_quad.png
backgroundPictureDir=/Volumes/${BASE_NAME}/.background
if [ -d ${backgroundPictureDir} ] ; then
  \rm -rf ${backgroundPictureDir}
fi
mkdir ${backgroundPictureDir}
cp ../Code/Gui/Main/images/${backgroundPictureName} ${backgroundPictureDir}

# We run applescript commands to set up the window properties.
# These properties are effectively set in the .DS_STORE folders in the image

echo "Running applescript"
applicationName=${OUTPUT_NAME}
echo '

   tell application "Finder"
     tell disk "'${BASE_NAME}'"
           open
           tell container window
             set current view to icon view
             set toolbar visible to false
             set statusbar visible to false
             set the bounds to {400, 100, 885, 430}             
           end tell
           set theViewOptions to the icon view options of container window
           set arrangement of theViewOptions to not arranged
           set icon size of theViewOptions to 72
           set background picture of theViewOptions to file ".background:'${backgroundPictureName}'" 
           make new alias file at container window to POSIX file "/Applications" with properties {name:"Applications"}
           set position of item "'${BASE_NAME}'" of container window to {50, 50}
           set position of item "Applications" of container window to {200, 50}     
           close
           open
           update without registering applications
           delay 5
     end tell
   end tell
   
' | osascript

sleep 4

# Fix permissions to make sure the whole thing isn't writable to others.
echo "Fixing permissions..."
chmod -Rf go-w /Volumes/${BASE_NAME}
sync
sync

# Unmount disk image
echo "Unmounting disk image..."
hdiutil detach ${device}

# Finalise (compress etc) package
echo "Finalising package"
hdiutil convert "${DIR_NAME}/${TMP_OUTPUT_DMG}" -format UDZO -imagekey zlib-level=9 -o "${DIR_NAME}/${OUTPUT_DMG}"

\rm -rf ${EFFECTIVE_ROOT}
\rm -f ${DIR_NAME}/${TMP_OUTPUT_DMG}
