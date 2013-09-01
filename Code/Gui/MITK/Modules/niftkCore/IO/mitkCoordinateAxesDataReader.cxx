/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkCoordinateAxesDataReader.h"
#include <itksys/SystemTools.hxx>
#include <vtkMatrix4x4.h>
#include <vtkSmartPointer.h>
#include <mitkFileIOUtils.h>

namespace mitk {

//-----------------------------------------------------------------------------
CoordinateAxesDataReader::CoordinateAxesDataReader()
: m_OutputCache(NULL)
{

}


//-----------------------------------------------------------------------------
CoordinateAxesDataReader::~CoordinateAxesDataReader()
{

}


//-----------------------------------------------------------------------------
void CoordinateAxesDataReader::Update()
{
  this->GenerateData();
}


//-----------------------------------------------------------------------------
const char* CoordinateAxesDataReader::GetFileName() const
{
  return m_FileName.c_str();
}


//-----------------------------------------------------------------------------
void CoordinateAxesDataReader::SetFileName(const char* aFileName)
{
  m_FileName = aFileName;
}


//-----------------------------------------------------------------------------
const char* CoordinateAxesDataReader::GetFilePrefix() const
{
  return m_FilePrefix.c_str();
}


//-----------------------------------------------------------------------------
void CoordinateAxesDataReader::SetFilePrefix(const char* aFilePrefix)
{
  m_FilePrefix = aFilePrefix;
}


//-----------------------------------------------------------------------------
const char* CoordinateAxesDataReader::GetFilePattern() const
{
  return m_FilePattern.c_str();
}


//-----------------------------------------------------------------------------
void CoordinateAxesDataReader::SetFilePattern(const char* aFilePattern)
{
  m_FilePattern = aFilePattern;
}


//-----------------------------------------------------------------------------
bool CoordinateAxesDataReader::CanReadFile(
  const std::string filename, const std::string /*filePrefix*/,
  const std::string /*filePattern*/)
{
  // First check the extension
  if(  filename == "" )
  {
    return false;
  }

  std::string ext = itksys::SystemTools::GetFilenameLastExtension(filename);
  ext = itksys::SystemTools::LowerCase(ext);

  if (ext == mitk::CoordinateAxesData::FILE_EXTENSION)
  {
    return true;
  }

  return false;
}


//-----------------------------------------------------------------------------
void CoordinateAxesDataReader::GenerateData()
{
  MITK_INFO << "Reading CoordinateAxesData";
  if (!m_OutputCache)
  {
    Superclass::SetNumberOfRequiredOutputs(0);
    this->GenerateOutputInformation();
  }

  if (!m_OutputCache)
  {
    itkWarningMacro("Cache is empty!");
  }

  Superclass::SetNumberOfRequiredOutputs(1);
  Superclass::SetNthOutput(0, m_OutputCache.GetPointer());
}


//-----------------------------------------------------------------------------
void CoordinateAxesDataReader::GenerateOutputInformation()
{
  m_OutputCache = OutputType::New();

  std::string ext = itksys::SystemTools::GetFilenameLastExtension(m_FileName);
  ext = itksys::SystemTools::LowerCase(ext);

  if ( m_FileName == "")
  {
    MITK_ERROR << "CoordinateAxesDataReader:No file name specified.";
  }
  else if (ext == mitk::CoordinateAxesData::FILE_EXTENSION)
  {
    try
    {
      vtkSmartPointer<vtkMatrix4x4> matrix = mitk::LoadVtkMatrix4x4FromFile(m_FileName);
      if (matrix == NULL)
      {
        MITK_ERROR << "Programming error: Current spec is that mitk::LoadVtkMatrix4x4FromFile must always return a matrix." << std::endl;
        return;
      }

      m_OutputCache->SetVtkMatrix(*matrix);
    }
    catch (mitk::Exception e)
    {
      MITK_ERROR << e.GetDescription();
    }
    catch(...)
    {
      MITK_ERROR << "Unknown error occured while trying to read file.";
    }
  }
}


//-----------------------------------------------------------------------------
BaseDataSource::DataObjectPointer CoordinateAxesDataReader::MakeOutput ( DataObjectPointerArraySizeType idx )
{
  return OutputType::New().GetPointer();
}


//-----------------------------------------------------------------------------
BaseDataSource::DataObjectPointer CoordinateAxesDataReader::MakeOutput(const DataObjectIdentifierType& name)
{
  itkDebugMacro("MakeOutput(" << name << ")");
  if( this->IsIndexedOutputName(name) )
    {
    return this->MakeOutput( this->MakeIndexFromOutputName(name) );
    }
  return static_cast<itk::DataObject*>(OutputType::New().GetPointer());

}


} // end namespace
