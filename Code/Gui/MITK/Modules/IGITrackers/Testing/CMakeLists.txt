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

MITK_CREATE_MODULE_TESTS(LABELS niftkIGITrackers)

if(TARGET ${TESTDRIVER})

  mitk_use_modules(TARGET ${TESTDRIVER} PACKAGES )

#mitkAddCustomModuleTest(NDIPolarisTest niftkNDIConnectionTest Polaris ${NIFTK_DATA_DIR}/Input/SmartLiverSurgeryPolaris.IGTToolStorage 4)
#mitkAddCustomModuleTest(NDIAuroraTest  niftkNDIConnectionTest Aurora  ${NIFTK_DATA_DIR}/Input/SmartLiverSurgeryAurora.IGTToolStorage 3)

endif()
