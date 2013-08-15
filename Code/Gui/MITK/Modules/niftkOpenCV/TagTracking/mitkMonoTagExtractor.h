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
#include <vtkMatrix4x4.h>

namespace mitk {

class MonoTagExtractorPrivate;

/**
 * \class MonoTagExtractor
 * \brief Takes a single RGBA video image, and uses ARUCO to extract tag positions.
 *
 * This class is more of a test class, as most use-cases in NifTK use the
 * stereo version in order to do triangulation.
 */
class NIFTKOPENCV_EXPORT MonoTagExtractor : public itk::Object
{

public:

  mitkClassMacro(MonoTagExtractor, itk::Object);
  itkNewMacro(MonoTagExtractor);

  /**
   * \brief Pass in an image, and then tag positions are extracted and returned in the provided mitk::PointSet.
   * \param image RGBA colour image
   * \param minSize the minimum size of the tag, measured as a fraction between 0 and 1 of the maximum of the number of rows and columns.
   * \param maxSize the maximum size of the tag, measured as a fraction between 0 and 1 of the maximum of the number of rows and columns.
   * \param blockSize window size for adaptive thresholding
   * \param offset the amount below the mean intensity of the window to set the threshold at
   * \param pointSet a point set object, allocated outside of this method. i.e. pointer must be non-null when calling this method.
   * \param cameraToWorld if not null, all reconstructed points are multiplied by this transform.
   */
  void ExtractPoints(const mitk::Image::Pointer image,
                     const float& minSize,
                     const float& maxSize,
                     const int& blockSize,
                     const int& offset,
                     const vtkMatrix4x4* cameraToWorld,
                     mitk::PointSet::Pointer pointSet
                     );

protected:

  MonoTagExtractor();
  virtual ~MonoTagExtractor();

  MonoTagExtractor(const MonoTagExtractor&); // Purposefully not implemented.
  MonoTagExtractor& operator=(const MonoTagExtractor&); // Purposefully not implemented.

private:

  mitk::MonoTagExtractorPrivate* m_PIMPL;

}; // end class

} // end namespace

#endif
