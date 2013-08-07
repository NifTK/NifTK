/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkUltrasoundPinCalibration_h
#define mitkUltrasoundPinCalibration_h

#include "niftkOpenCVExports.h"
#include <string>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkCommon.h>
#include <mitkVector.h>
#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>

namespace mitk {

/**
 * \class UltrasoundPinCalibration
 * \brief Does an ultrasound probe calibration from an ordered list of tracker matrices, and pin locations (x,y pixels).
 */
class NIFTKOPENCV_EXPORT UltrasoundPinCalibration : public itk::Object
{

public:

  mitkClassMacro(UltrasoundPinCalibration, itk::Object);
  itkNewMacro(UltrasoundPinCalibration);

  bool CalibrateUsingTrackerPointAndFilesInTwoDirectories(
      const std::string& matrixDirectory,
      const std::string& pointDirectory,
      const std::string& outputFileName,
      const mitk::Point3D& invariantPoint,
      const mitk::Point2D& originInImagePlaneInPixels,
      double &residualError
      );

  bool CalibrateUsingTrackerPoint(
      const std::vector< vtkSmartPointer<vtkMatrix4x4> >& matrices,
      const std::vector<mitk::Point3D>& points,
      const mitk::Point3D& invariantPoint,
      const mitk::Point2D& originInImagePlaneInPixels,
      double &residualError,
      vtkMatrix4x4 &outputMatrix
      );

  bool Calibrate(
      const std::vector< vtkSmartPointer<vtkMatrix4x4> >& matrices,
      const std::vector<mitk::Point3D>& points,
      const vtkMatrix4x4& worldToPhantomMatrix,
      const mitk::Point3D& invariantPoint,
      const mitk::Point2D& originInImagePlaneInPixels,
      double &residualError,
      vtkMatrix4x4 &outputMatrix
      );

protected:

  UltrasoundPinCalibration();
  virtual ~UltrasoundPinCalibration();

  UltrasoundPinCalibration(const UltrasoundPinCalibration&); // Purposefully not implemented.
  UltrasoundPinCalibration& operator=(const UltrasoundPinCalibration&); // Purposefully not implemented.

private:

}; // end class

} // end namespace

#endif
