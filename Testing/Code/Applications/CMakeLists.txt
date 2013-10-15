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

# Example test, documented in NifTK/Doc/TechnicalManual/TestingApplications.dox

add_test(niftkAdd-Run2D ${EXECUTABLE_OUTPUT_PATH}/niftkAdd -i ${INPUT_DATA}/cte_circle_gm.png -j ${INPUT_DATA}/cte_circle_wm.png -o ${TEMP}/niftkAdd-2D-output.png)
add_test(niftkAdd-Test2D ${EXECUTABLE_OUTPUT_PATH}/niftkTestCompareImage -i ${TEMP}/niftkAdd-2D-output.png -j ${BASELINE}/niftkAdd-2D-output.png -intensity )

# Additional image checks can be added to niftkTestCompareImage to make it more useful.
# Additional ctest tests should be added below, eventually covering all the applications in NifTK/Code/Applications.
