/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkStereoTagExtractor_h
#define mitkStereoTagExtractor_h

#include "niftkOpenCVExports.h"
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkCommon.h>
#include <mitkImage.h>
#include <mitkPointSet.h>
#include <cv.h>
#include <vtkMatrix4x4.h>

namespace mitk {

/**
 * \class StereoTagExtractor
 * \brief Command object to take a stereo pair of images and camera params, and extract tag positions using triangulation.
 */
class NIFTKOPENCV_EXPORT StereoTagExtractor : public itk::Object
{

public:

  mitkClassMacro(StereoTagExtractor, itk::Object);
  itkNewMacro(StereoTagExtractor);

  /**
   * \brief Pass in a stereo pair of images, and tag positions are extracted and returned in the provided mitk::PointSet.
   * \param leftImage RGB colour image
   * \param rightImage RGB colour image
   * \param minSize the minimum size of the tag, measured as a fraction between 0 and 1 of the maximum of the number of rows and columns.
   * \param maxSize the maximum size of the tag, measured as a fraction between 0 and 1 of the maximum of the number of rows and columns.
   * \param leftCameraIntrinsics the camera intrinsic params, as calculated by the camera calibration routines.
   * \param rightCameraIntrinsics the camera intrinsic params, as calculated by the camera calibration routines.
   * \param rightToLeftRotationVector a [1x3] rotation vector as per the Rodrigues formulation.
   * \param rightToLeftTranslationVector a [1x3] translation vector.
   * \param cameraToWorld if not null, all reconstructed points are multiplied by this transform.
   */
  void ExtractPoints(const mitk::Image::Pointer leftImage,
                     const mitk::Image::Pointer rightImage,
                     const float& minSize,
                     const float& maxSize,
                     const CvMat& leftCameraIntrinsics,
                     const CvMat& rightCameraIntrinsics,
                     const CvMat& rightToLeftRotationVector,
                     const CvMat& rightToLeftTranslationVector,
                     mitk::PointSet::Pointer pointSet,
                     const vtkMatrix4x4* cameraToWorld
                     );

  /**
   * \brief Overloaded interface for other method, extracting the necessary matrices off of the mitk::Image
   */
  void ExtractPoints(const mitk::Image::Pointer leftImage,
                     const mitk::Image::Pointer rightImage,
                     const float& minSize,
                     const float& maxSize,
                     mitk::PointSet::Pointer pointSet,
                     const vtkMatrix4x4* cameraToWorld
                     );

protected:

  StereoTagExtractor();
  virtual ~StereoTagExtractor();

  StereoTagExtractor(const StereoTagExtractor&); // Purposefully not implemented.
  StereoTagExtractor& operator=(const StereoTagExtractor&); // Purposefully not implemented.

}; // end class

} // end namespace

#endif // mitkStereoTagExtractor_h
