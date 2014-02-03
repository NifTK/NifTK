/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkUltrasoundCalibration_h
#define mitkUltrasoundCalibration_h

#include "niftkOpenCVExports.h"
#include <string>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkCommon.h>
#include <mitkVector.h>
#include <vtkMatrix4x4.h>
#include <vtkSmartPointer.h>
#include <cv.h>

namespace mitk {

/**
 * \class UltrasoundCalibration
 * \brief Base class for mitk::UltrasoundPinCalibration and mitk::UltrasoundPointerCalibration.
 */
class NIFTKOPENCV_EXPORT UltrasoundCalibration : public itk::Object
{

public:

  mitkClassMacro(UltrasoundCalibration, itk::Object);

  itkSetMacro(MillimetresPerPixel, mitk::Point2D);
  itkGetMacro(MillimetresPerPixel, mitk::Point2D);

  itkSetMacro(OptimiseScaling, bool);
  itkGetMacro(OptimiseScaling, bool);

  void InitialiseMillimetresPerPixel(const std::vector<float>& commandLineArgs);

  void InitialiseInitialGuess(const std::string& fileName);

  void SetInitialGuess(const vtkMatrix4x4& matrix);

  /**
   * \brief Method that provides directory scanning before calling the other calibrate method.
   *
   * More specifically, it will look in 2 directories, where 1 directory contains JUST the tracking
   * matrices, with each matrix is a plain text file of 4 rows of 4 columns and the second directory
   * contains JUST the points, each point in a separate plain text file containing a single x, y coordinate.
   * The filenames are used for sorting so that there must be the same number of files in each directory,
   * and the sort order must correspond.
   *
   * \param[In] matrixDirectory directory containing tracking matrices
   * \param[In] pointDirectory directory containing 2D pixel locations of a pin-head.
   * \param[Out] outputMatrix the output transformation.
   * \return the RMS residual error
   */
  double CalibrateFromDirectories(
      const std::string& matrixDirectory,
      const std::string& pointDirectory,
      vtkMatrix4x4& outputMatrix
      );

  /**
   * \brief Derived classes implement this calibration method.
   * \param[In] matrices a vector of 4x4 matrices representing rigid body tracking transformation.
   * \param[In] points a vector of 2D pixel locations in the same order as the tracking transformations.
   * \param[Out] outputMatrix the calibration matrix
   * \return the RMS residual error
   */
  virtual double Calibrate(
      const std::vector< cv::Mat >& matrices,
      const std::vector< std::pair<int, cv::Point2d> >& points,
      cv::Matx44d& outputMatrix
      ) = 0;

protected:

  UltrasoundCalibration();
  virtual ~UltrasoundCalibration();

  UltrasoundCalibration(const UltrasoundCalibration&); // Purposefully not implemented.
  UltrasoundCalibration& operator=(const UltrasoundCalibration&); // Purposefully not implemented.

protected:

  mitk::Point2D       m_MillimetresPerPixel;
  bool                m_OptimiseScaling;
  std::vector<double> m_InitialGuess;

}; // end class

} // end namespace

#endif
