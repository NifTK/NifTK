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

#include <cstdlib>
#include "mitkCameraCalibrationFromDirectory.h"

int main(int argc, char** argv)
{
  std::string directoryName = argv[1];
  std::string outputFile = "calib.txt";

  mitk::CameraCalibrationFromDirectory::Pointer calibrationObject = mitk::CameraCalibrationFromDirectory::New();
  //calibrationObject->Calibrate(directoryName, 16, 12, 3.5, outputFile);
  calibrationObject->Calibrate(directoryName, 14, 10, 3, outputFile);

  return EXIT_SUCCESS;
}
