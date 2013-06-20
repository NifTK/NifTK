/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef CoordinateAxesDataReaderFactory_h
#define CoordinateAxesDataReaderFactory_h

#ifdef _MSC_VER
#pragma warning ( disable : 4786 )
#endif

#include "niftkCoreExports.h"
#include <itkObjectFactoryBase.h>
#include <mitkBaseData.h>

namespace mitk
{
/**
 * \class CoordinateAxesDataReaderFactory
 * \brief Create instances of CoordinateAxesDataReader objects using an object factory pattern.
 */
class NIFTKCORE_EXPORT CoordinateAxesDataReaderFactory : public itk::ObjectFactoryBase
{
public:

  /** Standard class typedefs. */
  typedef CoordinateAxesDataReaderFactory Self;
  typedef itk::ObjectFactoryBase          Superclass;
  typedef itk::SmartPointer<Self>         Pointer;
  typedef itk::SmartPointer<const Self>   ConstPointer;

  /** Class methods used to interface with the registered factories. */
  virtual const char* GetITKSourceVersion(void) const;
  virtual const char* GetDescription(void) const;

  /** Method for class instantiation. */
  itkFactorylessNewMacro(Self);
  static CoordinateAxesDataReaderFactory* FactoryNew() { return new CoordinateAxesDataReaderFactory;}

  /** Run-time type information (and related methods). */
  itkTypeMacro(CoordinateAxesDataReaderFactory, ObjectFactoryBase);

  /** Register one factory of this type  */
  static void RegisterOneFactory(void)
  {
    CoordinateAxesDataReaderFactory::Pointer factory = CoordinateAxesDataReaderFactory::New();
    ObjectFactoryBase::RegisterFactory(factory);
  }

protected:
  CoordinateAxesDataReaderFactory(); // Purposefully hidden.
  ~CoordinateAxesDataReaderFactory(); // Purposefully hidden.

private:
  CoordinateAxesDataReaderFactory(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
};

} // end namespace mitk

#endif // CoordinateAxesDataReaderFactory_h
