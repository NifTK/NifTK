/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef MITKOPENCVTEST_H
#define MITKOPENCVTEST_H

#include "niftkOpenCVExports.h"
#include <string>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkCommon.h>

namespace mitk {

/**
 * \class OpenCVTest
 * \brief Test Class
 */
class NIFTKOPENCV_EXPORT OpenCVTest : public itk::Object
{

public:

  mitkClassMacro(OpenCVTest, itk::Object);
  itkNewMacro(OpenCVTest);

  void Run(const std::string& fileName);

protected:

  OpenCVTest();
  virtual ~OpenCVTest();

  OpenCVTest(const OpenCVTest&); // Purposefully not implemented.
  OpenCVTest& operator=(const OpenCVTest&); // Purposefully not implemented.

}; // end class

} // end namespace

#endif
