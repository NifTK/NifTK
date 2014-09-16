/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkCoordinateAxesDataWriterFactory_h
#define mitkCoordinateAxesDataWriterFactory_h

#include "niftkCoreExports.h"
#include <itkObjectFactoryBase.h>
#include <mitkCoreObjectFactoryBase.h>
#include <mitkBaseData.h>

namespace mitk
{

/**
 * \class CoordinateAxesDataWriterFactory
 * \brief Factory to create CoordinateAxesDataWriter.
 */
class NIFTKCORE_EXPORT CoordinateAxesDataWriterFactory : public itk::ObjectFactoryBase
{
public:

  mitkClassMacro( mitk::CoordinateAxesDataWriterFactory, itk::ObjectFactoryBase )

  /** Class methods used to interface with the registered factories. */
  virtual const char* GetITKSourceVersion(void) const;
  virtual const char* GetDescription(void) const;

  /** Method for class instantiation. */
  itkFactorylessNewMacro(Self);

protected:
  CoordinateAxesDataWriterFactory(); // Purposefully hidden.
  ~CoordinateAxesDataWriterFactory(); // Purposefully hidden.

private:
  CoordinateAxesDataWriterFactory(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};

} // end namespace mitk

#endif // CoordinateAxesDataWriterFactory_h



