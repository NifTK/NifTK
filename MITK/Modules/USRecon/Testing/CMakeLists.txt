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

MITK_CREATE_MODULE_TESTS(LABELS niftkUSRecon)

if(TARGET ${TESTDRIVER})

  mitk_use_modules(TARGET ${TESTDRIVER} PACKAGES OpenCV)

  add_test(USRecon-1 ${CXX_TEST_PATH}/niftkUSReconTestDriver
           niftkUSReconTest
           # Add extra args here
          )

    # Add extra unit tests here

endif()

