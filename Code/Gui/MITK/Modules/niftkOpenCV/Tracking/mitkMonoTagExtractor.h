/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkMonoTagExtractor_h
#define mitkMonoTagExtractor_h

#include "niftkOpenCVExports.h"
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkCommon.h>
#include <mitkImage.h>
#include <mitkPointSet.h>
#include <cv.h>

namespace mitk {

/**
 * \class MonoTagExtractor
 * \brief Command Object to take a single image and camera params, and extract tag positions.
 */
class NIFTKOPENCV_EXPORT MonoTagExtractor : public itk::Object
{

public:

  mitkClassMacro(MonoTagExtractor, itk::Object);
  itkNewMacro(MonoTagExtractor);

  /**
   * \brief Pass in an image, and tag positions are extracted and returned in the provided mitk::PointSet.
   * \param image RGB colour image
   * \param minSize the minimum size of the tag, measured as a fraction between 0 and 1 of the maximum of the number of rows and columns.
   * \param maxSize the maximum size of the tag, measured as a fraction between 0 and 1 of the maximum of the number of rows and columns.
   * \param pointSet a point set object, allocated outside of this method. i.e. pointer must be non-null when calling this method.
   */
  void ExtractPoints(const mitk::Image::Pointer image,
                     const float& minSize,
                     const float& maxSize,
                     mitk::PointSet::Pointer pointSet
                     );

protected:

  MonoTagExtractor();
  virtual ~MonoTagExtractor();

  MonoTagExtractor(const MonoTagExtractor&); // Purposefully not implemented.
  MonoTagExtractor& operator=(const MonoTagExtractor&); // Purposefully not implemented.

}; // end class

} // end namespace

#endif // mitkMonoTagExtractor_h
