/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
#ifndef niftkEdgeDetectionExampleLauncher_h
#define niftkEdgeDetectionExampleLauncher_h

#include "niftkCUDAExports.h"
#include <mitkDataNode.h>
#include <mitkBaseRenderer.h>
#include <mitkDataStorage.h>

namespace niftk
{

/**
* \brief Runs edge detection on an MITK DataNode containing a niftk::CUDAImage.
* \see niftk::RunEdgeDetectionKernel
*/
void NIFTKCUDA_EXPORT EdgeDetectionExampleLauncher(mitk::DataStorage* dataStorage, mitk::DataNode* node,
                                                   const mitk::BaseRenderer* renderer);

} // end namespace

#endif
