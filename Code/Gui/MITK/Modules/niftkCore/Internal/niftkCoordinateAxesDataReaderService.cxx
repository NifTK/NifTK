/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkCoordinateAxesDataReaderService.h"
#include "niftkCoreIOMimeTypes.h"
#include <mitkCoordinateAxesData.h>
#include <mitkFileIOUtils.h>
#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>

#include <iostream>
#include <fstream>
#include <locale>

namespace niftk {

//-----------------------------------------------------------------------------
CoordinateAxesDataReaderService::CoordinateAxesDataReaderService()
  : AbstractFileReader(
      mitk::CustomMimeType(niftk::CoreIOMimeTypes::TRANSFORM4X4_MIMETYPE_NAME()),
      "NifTK Coordinate Axes Reader")
{
  RegisterService();
}


//-----------------------------------------------------------------------------
CoordinateAxesDataReaderService::CoordinateAxesDataReaderService(const CoordinateAxesDataReaderService& other)
  : mitk::AbstractFileReader(other)
{
}


//-----------------------------------------------------------------------------
CoordinateAxesDataReaderService::~CoordinateAxesDataReaderService()
{}


//-----------------------------------------------------------------------------
std::vector< itk::SmartPointer<mitk::BaseData> > CoordinateAxesDataReaderService::Read()
{
  std::locale::global(std::locale("C"));
  std::vector< itk::SmartPointer<mitk::BaseData> > result;

  std::string fileName = this->GetInputLocation();
  MITK_INFO << "Reading .4x4 transform from:" << fileName << std::endl;

  vtkSmartPointer<vtkMatrix4x4> matrix = mitk::LoadVtkMatrix4x4FromFile(fileName);
  mitk::CoordinateAxesData::Pointer transform = mitk::CoordinateAxesData::New();
  transform->SetVtkMatrix(*matrix);

  result.push_back(itk::SmartPointer<mitk::BaseData>(transform));
  return result;
}


//-----------------------------------------------------------------------------
CoordinateAxesDataReaderService* CoordinateAxesDataReaderService::Clone() const
{
  return new CoordinateAxesDataReaderService(*this);
}

} // end namespace mitk
