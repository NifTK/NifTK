/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkPCLTest_h
#define mitkPCLTest_h

#include "niftkIGIExports.h"
#include <mitkCommon.h>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <itkObjectFactoryBase.h>

namespace mitk {

/**
 * \class PCLTest
 * \brief Command used to write points to file.
 */
class NIFTKIGI_EXPORT PCLTest : public itk::Object
{
public:

  mitkClassMacro(PCLTest, itk::Object);
  itkNewMacro(PCLTest);

  /**
   * \brief Write My Documentation
   */
  void Update(const std::string& fileName);

protected:

  PCLTest(); // Purposefully hidden.
  virtual ~PCLTest(); // Purposefully hidden.

  PCLTest(const PCLTest&); // Purposefully not implemented.
  PCLTest& operator=(const PCLTest&); // Purposefully not implemented.

private:

}; // end class

} // end namespace

#endif
