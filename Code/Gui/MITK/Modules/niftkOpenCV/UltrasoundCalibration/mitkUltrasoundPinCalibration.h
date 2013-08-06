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

  /**
   * \brief Calibration function that returns the residual (basically the spread of the reconstructed points).
   */
  double Calibrate(const std::string& matrixDirectory,
      const std::string& pointDirectory,
      const std::string& outputFileName
      );

protected:

  UltrasoundPinCalibration();
  virtual ~UltrasoundPinCalibration();

  UltrasoundPinCalibration(const UltrasoundPinCalibration&); // Purposefully not implemented.
  UltrasoundPinCalibration& operator=(const UltrasoundPinCalibration&); // Purposefully not implemented.

}; // end class

} // end namespace

#endif
