/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date$
 Revision          : $Revision$
 Last modified by  : $Author$

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef MITKCAMERACALIBRATIONFROMDIRECTORY_H
#define MITKCAMERACALIBRATIONFROMDIRECTORY_H

#include "niftkOpenCVExports.h"
#include <itkObject.h>
#include <mitkDataStorage.h>

namespace mitk {

/**
 * \class CameraCalibrationFromDirectory
 * \brief Does a camera calibration from a directory containing a list of image files.
 */
class NIFTKOPENCV_EXPORT CameraCalibrationFromDirectory : public itk::Object
{

public:

  mitkClassMacro(CameraCalibrationFromDirectory, itk::Object);
  itkNewMacro(CameraCalibrationFromDirectory);

  bool Calibrate(const std::string& fullDirectoryName,
      const int& numberCornersX,
      const int& numberCornersY,
      const float& sizeSquareMillimeters,
      const std::string& outputFile
      );

protected:

  CameraCalibrationFromDirectory();
  virtual ~CameraCalibrationFromDirectory();

  CameraCalibrationFromDirectory(const CameraCalibrationFromDirectory&); // Purposefully not implemented.
  CameraCalibrationFromDirectory& operator=(const CameraCalibrationFromDirectory&); // Purposefully not implemented.

private:

}; // end class

} // end namespace

#endif
