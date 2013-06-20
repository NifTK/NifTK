/*===================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center,
Division of Medical and Biological Informatics.
All rights reserved.

This software is distributed WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR
A PARTICULAR PURPOSE.

See LICENSE.txt or http://www.mitk.org for details.

===================================================================*/

#include "mitkCoordinateAxesDataWriter.h"
#include <mitkCoordinateAxesData.h>
#include <itksys/SystemTools.hxx>
#include <mitkFileIOUtils.h>
#include <vtkMatrix4x4.h>
#include <vtkSmartPointer.h>

namespace mitk
{

//-----------------------------------------------------------------------------
CoordinateAxesDataWriter::CoordinateAxesDataWriter()
: m_FileName("")
, m_FilePrefix("")
, m_FilePattern("")
, m_Success(false)
{
  this->SetNumberOfRequiredInputs( 1 );
}


//-----------------------------------------------------------------------------
CoordinateAxesDataWriter::~CoordinateAxesDataWriter()
{
}


//-----------------------------------------------------------------------------
void CoordinateAxesDataWriter::Update()
{
  Write();
}


//-----------------------------------------------------------------------------
void CoordinateAxesDataWriter::Write()
{
  if ( this->GetInput() == NULL )
  {
    itkExceptionMacro(<<"Write:Please specify an input!");
    return;
  }

  this->UpdateOutputInformation();
  (*(this->GetInputs().begin()))->SetRequestedRegionToLargestPossibleRegion();
  this->PropagateRequestedRegion(NULL);
  this->UpdateOutputData(NULL);
}


//-----------------------------------------------------------------------------
bool CoordinateAxesDataWriter::CanWriteBaseDataType(BaseData::Pointer data)
{
  return (dynamic_cast<mitk::CoordinateAxesData*>(data.GetPointer()) != NULL);
}


//-----------------------------------------------------------------------------
void CoordinateAxesDataWriter::DoWrite(BaseData::Pointer data)
{
  if (CanWriteBaseDataType(data))
  {
    this->SetInputCoordinateAxesData(dynamic_cast<mitk::CoordinateAxesData*>(data.GetPointer()));
    this->Update();
  }
}


//-----------------------------------------------------------------------------
void CoordinateAxesDataWriter::GenerateData()
{
  m_Success = false;
  InputType* input = this->GetInput();

  if (input == NULL)
  {
    itkWarningMacro(<<"Sorry, input to CoordinateAxesDataWriter is NULL!");
    return;
  }
  if ( m_FileName == "" )
  {
    itkWarningMacro( << "Sorry, filename has not been set!" );
    return ;
  }

  std::string ext = itksys::SystemTools::GetFilenameLastExtension(m_FileName);
  ext = itksys::SystemTools::LowerCase(ext);

  if (ext == mitk::CoordinateAxesData::FILE_EXTENSION)
  {

    MITK_INFO << "Writing CoordinateAxesData to " << m_FileName;

    vtkSmartPointer<vtkMatrix4x4> matrix = vtkMatrix4x4::New();
    input->GetVtkMatrix(*matrix);

    m_Success = mitk::SaveVtkMatrix4x4ToFile(m_FileName, *matrix);

    MITK_INFO << "CoordinateAxesData written";
  }
}


//-----------------------------------------------------------------------------
void CoordinateAxesDataWriter::SetInputCoordinateAxesData( InputType* data )
{
  this->ProcessObject::SetNthInput( 0, data );
}


//-----------------------------------------------------------------------------
mitk::CoordinateAxesDataWriter::InputType* CoordinateAxesDataWriter::GetInput()
{
  if ( this->GetNumberOfInputs() < 1 )
  {
    return NULL;
  }
  else
  {
    return dynamic_cast<InputType*> ( this->ProcessObject::GetInput( 0 ) );
  }
}


//-----------------------------------------------------------------------------
std::vector<std::string> CoordinateAxesDataWriter::GetPossibleFileExtensions()
{
  std::vector<std::string> possibleFileExtensions;
  possibleFileExtensions.push_back(mitk::CoordinateAxesData::FILE_EXTENSION);
  return possibleFileExtensions;
}

} // end namespace

