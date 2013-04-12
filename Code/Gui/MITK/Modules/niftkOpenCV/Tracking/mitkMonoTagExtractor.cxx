/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkMonoTagExtractor.h"

namespace {

//-----------------------------------------------------------------------------
MonoTagExtractor::MonoTagExtractor()
{

}


//-----------------------------------------------------------------------------
MonoTagExtractor::~MonoTagExtractor()
{

}


//-----------------------------------------------------------------------------
void MonoTagExtractor::ExtractPoints(const mitk::Image::Pointer image,
                                     const float& minSize,
                                     const float& maxSize,
                                     const CvMat* cameraIntrinsics,
                                     mitk::PointSet::Pointer pointSet
                                    )
{

}

//-----------------------------------------------------------------------------
} // end namespace
