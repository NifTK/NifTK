/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef CoordinateAxesDataWriterFactory_h
#define CoordinateAxesDataWriterFactory_h

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

  /** Register one factory of this type  */
  static void RegisterOneFactory(void)
  {
    static bool IsRegistered = false;
    if ( !IsRegistered )
    {
      CoordinateAxesDataWriterFactory::Pointer factory = CoordinateAxesDataWriterFactory::New();
      ObjectFactoryBase::RegisterFactory( factory );
      IsRegistered = true;
    }
  }

protected:
  CoordinateAxesDataWriterFactory(); // Purposefully hidden.
  ~CoordinateAxesDataWriterFactory(); // Purposefully hidden.

private:
  CoordinateAxesDataWriterFactory(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};

} // end namespace mitk

#endif // CoordinateAxesDataWriterFactory_h



