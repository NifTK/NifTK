/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkAddTwoIntegers_h
#define niftkAddTwoIntegers_h

#include <niftkCUDAKernelsWin32ExportHeader.h>

namespace niftk
{

/**
* \brief Pointless minimal function, mainly just to test if anything at all could be run on a device.
*/
int NIFTKCUDAKERNELS_WINEXPORT AddTwoIntegers(int a, int b);

} // end namespace

#endif
