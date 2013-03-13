/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef MITKTESTLINEEXTRACTION_H
#define MITKTESTLINEEXTRACTION_H

#include "niftkOpenCVExports.h"
#include <string>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkCommon.h>

namespace mitk {

/**
 * \class TestLineExtraction
 * \brief Test Class
 */
class NIFTKOPENCV_EXPORT TestLineExtraction : public itk::Object
{

public:

  mitkClassMacro(TestLineExtraction, itk::Object);
  itkNewMacro(TestLineExtraction);

  void Run(const std::string& fileName);

protected:

  TestLineExtraction();
  virtual ~TestLineExtraction();

  TestLineExtraction(const TestLineExtraction&); // Purposefully not implemented.
  TestLineExtraction& operator=(const TestLineExtraction&); // Purposefully not implemented.

}; // end class

} // end namespace

#endif
