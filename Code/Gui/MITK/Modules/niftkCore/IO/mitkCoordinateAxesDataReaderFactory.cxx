/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkCoordinateAxesDataReaderFactory.h"
#include "mitkCoordinateAxesDataReader.h"
#include <mitkIOAdapter.h>
#include <itkVersion.h>

namespace mitk
{

//-----------------------------------------------------------------------------
CoordinateAxesDataReaderFactory::CoordinateAxesDataReaderFactory()
{
  typedef CoordinateAxesDataReader CoordinateAxesDataReaderType;
  this->RegisterOverride("mitkIOAdapter",
                         "mitkCoordinateAxesDataReader",
                         "Coordinate Axes Data Reader",
                         1,
                         itk::CreateObjectFunction<IOAdapter<CoordinateAxesDataReaderType> >::New());
}


//-----------------------------------------------------------------------------
CoordinateAxesDataReaderFactory::~CoordinateAxesDataReaderFactory()
{
}


//-----------------------------------------------------------------------------
const char* CoordinateAxesDataReaderFactory::GetITKSourceVersion() const
{
  return ITK_SOURCE_VERSION;
}


//-----------------------------------------------------------------------------
const char* CoordinateAxesDataReaderFactory::GetDescription() const
{
  return "CoordinateAxesDataReaderFactory, allows the loading of Coordinate Axes Data";
}

} // end namespace mitk
