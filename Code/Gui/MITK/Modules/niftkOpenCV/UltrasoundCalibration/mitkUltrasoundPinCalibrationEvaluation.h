/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkUltrasoundPinCalibrationEvaluation_h
#define mitkUltrasoundPinCalibrationEvaluation_h

#include "niftkOpenCVExports.h"
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkCommon.h>
#include <mitkVector.h>

namespace mitk {

/**
 * \class UltrasoundPinCalibrationEvaluation
 * \brief Evaluates directories of points and tracking matrices against a known
 * gold standard invariant point.
 */
class NIFTKOPENCV_EXPORT UltrasoundPinCalibrationEvaluation : public itk::Object
{

public:

  mitkClassMacro(UltrasoundPinCalibrationEvaluation, itk::Object);
  itkNewMacro(UltrasoundPinCalibrationEvaluation);

  /**
   * \brief Does evaluation.
   */
  void Evaluate(
      const std::string& matrixDirectory,
      const std::string& pointDirectory,
      const mitk::Point3D& invariantPoint,
      const mitk::Point2D& millimetresPerPixel,
      const std::string& calibrationMatrix,
      const std::string& cameraToWorldMatrix
      );

protected:

  UltrasoundPinCalibrationEvaluation();
  virtual ~UltrasoundPinCalibrationEvaluation();

  UltrasoundPinCalibrationEvaluation(const UltrasoundPinCalibrationEvaluation&); // Purposefully not implemented.
  UltrasoundPinCalibrationEvaluation& operator=(const UltrasoundPinCalibrationEvaluation&); // Purposefully not implemented.

private:

}; // end class

} // end namespace

#endif
