/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkMakeGridOf2DImages_h
#define mitkMakeGridOf2DImages_h

#include "niftkOpenCVExports.h"
#include <string>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkCommon.h>

namespace mitk {

/**
 * \class MakeGridOf2DImages
 * \brief Tiles images together, e.g. Tiles Tags into a board of Tags.
 */
class NIFTKOPENCV_EXPORT MakeGridOf2DImages : public itk::Object
{

public:

  mitkClassMacro(MakeGridOf2DImages, itk::Object);
  itkNewMacro(MakeGridOf2DImages);

  void MakeGrid(const std::string &inputDirectory,
                const std::vector<int>& imageSize,
                const std::vector<int>& gridDimensions,
                const std::string &outputImageFile,
                const bool fillLengthWise
                );

protected:

  MakeGridOf2DImages();
  virtual ~MakeGridOf2DImages();

  MakeGridOf2DImages(const MakeGridOf2DImages&); // Purposefully not implemented.
  MakeGridOf2DImages& operator=(const MakeGridOf2DImages&); // Purposefully not implemented.

private:

}; // end class

} // end namespace

#endif
