/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkSurfTester_h
#define mitkSurfTester_h

#include "niftkOpenCVExports.h"
#include <string>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkCommon.h>

namespace mitk {

/**
 * \class SurfTester
 * \brief Just playing around with OpenCV SURF.
 */
class NIFTKOPENCV_EXPORT SurfTester : public itk::Object
{

public:

  mitkClassMacro(SurfTester, itk::Object);
  itkNewMacro(SurfTester);

  void RunSurf(const std::string& inputFileName, const std::string& outputFileName);

protected:

  SurfTester();
  virtual ~SurfTester();

  SurfTester(const SurfTester&); // Purposefully not implemented.
  SurfTester& operator=(const SurfTester&); // Purposefully not implemented.

private:

}; // end class

} // end namespace

#endif
