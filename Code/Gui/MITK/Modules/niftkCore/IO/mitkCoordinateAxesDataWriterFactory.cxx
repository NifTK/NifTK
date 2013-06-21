/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkCoordinateAxesDataWriterFactory.h"
#include "mitkCoordinateAxesDataWriter.h"
#include <itkCreateObjectFunction.h>
#include <itkVersion.h>

namespace mitk
{

  /**
   * \class CreateCoordinateAxesDataWriter
   * \brief Nested function class to create CoordinateAxesDataWriter.
   */
  template <class T>
  class CreateCoordinateAxesDataWriter : public itk::CreateObjectFunctionBase
  {
  public:

    /** Standard class typedefs. */
    typedef CreateCoordinateAxesDataWriter  Self;
    typedef itk::SmartPointer<Self>    Pointer;

    /** Methods from itk:LightObject. */
    itkFactorylessNewMacro(Self);
    LightObject::Pointer CreateObject() { typename T::Pointer p = T::New();
      p->Register();
      return p.GetPointer();
    }

  protected:
    CreateCoordinateAxesDataWriter() {}
    ~CreateCoordinateAxesDataWriter() {}

  private:
    CreateCoordinateAxesDataWriter(const Self&); //purposely not implemented
    void operator=(const Self&); //purposely not implemented
  };

//-----------------------------------------------------------------------------
CoordinateAxesDataWriterFactory::CoordinateAxesDataWriterFactory()
{
  this->RegisterOverride("IOWriter",
                         "CoordinateAxesDataWriter",
                         "CoordinateAxesData Writer",
                         1,
                         mitk::CreateCoordinateAxesDataWriter< mitk::CoordinateAxesDataWriter >::New());
}


//-----------------------------------------------------------------------------
CoordinateAxesDataWriterFactory::~CoordinateAxesDataWriterFactory()
{
}


//-----------------------------------------------------------------------------
const char* CoordinateAxesDataWriterFactory::GetITKSourceVersion() const
{
  return ITK_SOURCE_VERSION;
}


//-----------------------------------------------------------------------------
const char* CoordinateAxesDataWriterFactory::GetDescription() const
{
  return "CoordinateAxesDataWriterFactory";
}

//-----------------------------------------------------------------------------
} // end namespace mitk
