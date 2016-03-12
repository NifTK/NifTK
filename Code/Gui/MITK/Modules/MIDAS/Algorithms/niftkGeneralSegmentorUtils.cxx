/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkGeneralSegmentorUtils.h"

#include <mitkImageAccessByItk.h>

//-----------------------------------------------------------------------------
void niftk::GenerateOutlineFromBinaryImage(mitk::Image::Pointer image,
    int axisNumber,
    int sliceNumber,
    int projectedSliceNumber,
    mitk::ContourModelSet::Pointer outputContourSet
    )
{
  try
  {
    AccessFixedTypeByItk_n(image,
        niftk::ITKGenerateOutlineFromBinaryImage,
        (unsigned char),
        (3),
        (axisNumber,
         sliceNumber,
         projectedSliceNumber,
         outputContourSet
        )
      );
  }
  catch(const mitk::AccessByItkException& e)
  {
    MITK_ERROR << "Failed in niftk::ITKGenerateOutlineFromBinaryImage due to:" << e.what();
    outputContourSet->Clear();
  }
}
