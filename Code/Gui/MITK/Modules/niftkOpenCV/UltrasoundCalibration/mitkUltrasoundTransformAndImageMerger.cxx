/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkUltrasoundTransformAndImageMerger.h"
#include <mitkExceptionMacro.h>
#include <mitkOpenCVMaths.h>
#include <mitkOpenCVFileIOUtils.h>

namespace mitk {

//-----------------------------------------------------------------------------
UltrasoundTransformAndImageMerger::~UltrasoundTransformAndImageMerger()
{
}


//-----------------------------------------------------------------------------
UltrasoundTransformAndImageMerger::UltrasoundTransformAndImageMerger()
{
}


//-----------------------------------------------------------------------------
void UltrasoundTransformAndImageMerger::Merge(
    const std::string& inputMatrixDirectory,
    const std::string& inputImageDirectory,
    const std::string& outputFileName
    )
{
}


//-----------------------------------------------------------------------------
} // end namespace
