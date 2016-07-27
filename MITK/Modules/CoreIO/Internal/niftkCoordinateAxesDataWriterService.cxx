/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkCoordinateAxesDataWriterService.h"

#include <niftkCoordinateAxesData.h>

#include "niftkCoreIOMimeTypes.h"

namespace niftk
{

//-----------------------------------------------------------------------------
CoordinateAxesDataWriterService::CoordinateAxesDataWriterService()
: mitk::AbstractFileWriter(CoordinateAxesData::GetStaticNameOfClass(),
                           mitk::CustomMimeType(niftk::CoreIOMimeTypes::TRANSFORM4X4_MIMETYPE_NAME()),
                           "NifTK Coordinate Axes Writer")
{
  RegisterService();
}


//-----------------------------------------------------------------------------
CoordinateAxesDataWriterService::CoordinateAxesDataWriterService(const CoordinateAxesDataWriterService& other)
: AbstractFileWriter(other)
{
}


//-----------------------------------------------------------------------------
CoordinateAxesDataWriterService::~CoordinateAxesDataWriterService()
{
}


//-----------------------------------------------------------------------------
CoordinateAxesDataWriterService* CoordinateAxesDataWriterService::Clone() const
{
  return new CoordinateAxesDataWriterService(*this);
}


//-----------------------------------------------------------------------------
void CoordinateAxesDataWriterService::Write()
{
  std::string fileName = this->GetOutputLocation();

  CoordinateAxesData::ConstPointer transform = dynamic_cast<const CoordinateAxesData*>(this->GetInput());
  transform->SaveToFile(fileName);
}

}
