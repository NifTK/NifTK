/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkStereoTagExtractor.h"

namespace mitk {

//-----------------------------------------------------------------------------
StereoTagExtractor::StereoTagExtractor()
{

}


//-----------------------------------------------------------------------------
StereoTagExtractor::~StereoTagExtractor()
{

}


//-----------------------------------------------------------------------------
void StereoTagExtractor::ExtractPoints(const mitk::Image::Pointer leftImage,
                                       const mitk::Image::Pointer rightImage,
                                       const float& minSize,
                                       const float& maxSize,
                                       const CvMat& leftCameraIntrinsics,
                                       const CvMat& rightCameraIntrinsics,
                                       const CvMat& rightToLeftRotationVector,
                                       const CvMat& rightToLeftTranslationVector,
                                       mitk::PointSet::Pointer pointSet
                                      )
{

}


//-----------------------------------------------------------------------------
} // end namespace
