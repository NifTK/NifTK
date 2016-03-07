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

add_test(niftkAdd-Run2D ${EXECUTABLE_OUTPUT_PATH}/niftkAdd -i ${INPUT_DATA}/cte_circle_gm.png -j ${INPUT_DATA}/cte_circle_wm.png -o ${TEMP}/niftkAdd-2D-output.png)
add_test(niftkAdd-Test2D ${EXECUTABLE_OUTPUT_PATH}/niftkTestCompareImage -i ${TEMP}/niftkAdd-2D-output.png -j ${BASELINE}/niftkAdd-2D-output.png -intensity )
set_property(TEST niftkAdd-Test2D APPEND PROPERTY DEPENDS niftkAdd-Run2D)
