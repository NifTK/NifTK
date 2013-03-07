/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef MITKTESTCORNEREXTRACTION_H
#define MITKTESTCORNEREXTRACTION_H

#include "niftkOpenCVExports.h"
#include <string>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkCommon.h>

namespace mitk {

/**
 * \class TestCornerExtraction
 * \brief Test Class
 */
class NIFTKOPENCV_EXPORT TestCornerExtraction : public itk::Object
{

public:

  mitkClassMacro(TestCornerExtraction, itk::Object);
  itkNewMacro(TestCornerExtraction);

  void Run(const std::string& fileNameLeft, const std::string& fileNameRight);

protected:

  TestCornerExtraction();
  virtual ~TestCornerExtraction();

  TestCornerExtraction(const TestCornerExtraction&); // Purposefully not implemented.
  TestCornerExtraction& operator=(const TestCornerExtraction&); // Purposefully not implemented.

}; // end class

} // end namespace

#endif
