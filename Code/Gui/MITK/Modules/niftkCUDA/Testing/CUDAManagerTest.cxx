/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <CUDAManager/CUDAManager.h>
#include <mitkTestingMacros.h>


void Stuff()
{
  CUDAManager*    cm = CUDAManager::GetInstance();
}


int CUDAManagerTest(int /*argc*/, char* /*argv*/[])
{
  MITK_TEST_BEGIN("CUDAManagerTest");
  Stuff();
  MITK_TEST_END();

  return EXIT_SUCCESS;
}
