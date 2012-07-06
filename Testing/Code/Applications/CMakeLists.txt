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
#  Last Changed      : $LastChangedDate$ 
#  Revision          : $Revision$
#  Last modified by  : $Author$
#
#  Original author   : m.clarkson@ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/

# Example test, documented in NifTK/Doc/TechnicalManual/TestingApplications.dox

ADD_TEST(niftkAdd-Run2D ${EXECUTABLE_OUTPUT_PATH}/niftkAdd -i ${INPUT_DATA}/cte_circle_gm.png -j ${INPUT_DATA}/cte_circle_wm.png -o ${TEMP}/niftkAdd-2D-output.png)
ADD_TEST(niftkAdd-Test2D ${EXECUTABLE_OUTPUT_PATH}/niftkTestCompareImage -i ${TEMP}/niftkAdd-2D-output.png -j ${BASELINE}/niftkAdd-2D-output.png -intensity )

# Additional image checks can be added to niftkTestCompareImage to make it more useful.
# Additional ctest tests should be added below, eventually covering all the applications in NifTK/Code/Applications.
                                                                  