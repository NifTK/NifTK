/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkUltrasoundTransformAndImageMerger_h
#define mitkUltrasoundTransformAndImageMerger_h

#include "niftkOpenCVExports.h"
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkCommon.h>
#include <mitkVector.h>

namespace mitk {

/**
 * \class UltrasoundTransformAndImageMerger
 * \brief Merges a directory of images and tracking data into a .mhd file, according to PLUS specifications.
 */
class NIFTKOPENCV_EXPORT UltrasoundTransformAndImageMerger : public itk::Object
{

public:

  mitkClassMacro(UltrasoundTransformAndImageMerger, itk::Object);
  itkNewMacro(UltrasoundTransformAndImageMerger);

  /**
   * \brief Does merging.
   */
  void Merge(
      const std::string& inputMatrixDirectory,
      const std::string& inputImageDirectory,
      const std::string& outputImageFileName,
      const std::string& outputDataFileName
      );

protected:

  UltrasoundTransformAndImageMerger();
  virtual ~UltrasoundTransformAndImageMerger();

  UltrasoundTransformAndImageMerger(const UltrasoundTransformAndImageMerger&); // Purposefully not implemented.
  UltrasoundTransformAndImageMerger& operator=(const UltrasoundTransformAndImageMerger&); // Purposefully not implemented.

private:

}; // end class

} // end namespace

#endif
