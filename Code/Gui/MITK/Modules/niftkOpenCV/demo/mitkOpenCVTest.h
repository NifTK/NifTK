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
