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

MITK_CREATE_MODULE_TESTS(LABELS niftkIGIDataSources)

#mitkAddCustomModuleTest(QmitkIGIUltrasonixToolMemoryTest-greyscale QmitkIGIUltrasonixToolMemoryTest ${NIFTK_DATA_DIR}/Input/cte_circle_gm.png)
#mitkAddCustomModuleTest(QmitkIGIUltrasonixToolMemoryTest-colour QmitkIGIUltrasonixToolMemoryTest ${NIFTK_DATA_DIR}/Baseline/mitkCoordinateAxesDataRenderingTestBaseline.png)

mitk_use_modules(TARGET ${TESTDRIVER}
  PACKAGES Qt4|QtCore Qt5|Core 
)
