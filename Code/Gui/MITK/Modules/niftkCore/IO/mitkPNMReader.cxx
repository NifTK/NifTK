/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkPNMReader.h"
#include "../Internal/niftkCoreIOMimeTypes.h"

#include <mitkCustomMimeType.h>
#include <mitkLogMacros.h>
#include <mitkImage.h>

#include <itksys/SystemTools.hxx>

#include <vtkSmartPointer.h>
#include <vtkPNMReader.h>
#include <vtkPNMWriter.h>
#include <vtkImageData.h>


mitk::PNMReader::PNMReader()
  : mitk::AbstractFileReader(CustomMimeType( niftk::CoreIOMimeTypes::PNM_MIMETYPE_NAME() ), niftk::CoreIOMimeTypes::PNM_MIMETYPE_DESCRIPTION() )
{
  m_ServiceReg = this->RegisterService();
}

mitk::PNMReader::PNMReader(const PNMReader &other)
  :mitk::AbstractFileReader(other)
{
}

mitk::PNMReader * mitk::PNMReader::Clone() const
{
  return new PNMReader(*this);
}


std::vector<itk::SmartPointer<mitk::BaseData> > mitk::PNMReader::Read()
{

  std::vector<itk::SmartPointer<mitk::BaseData> > result;
  try
  {
    const std::string& locale = "C";
    const std::string& currLocale = setlocale( LC_ALL, NULL );
    setlocale(LC_ALL, locale.c_str());

    std::string filename = this->GetInputLocation();

    std::string ext = itksys::SystemTools::GetFilenameLastExtension(filename);
    ext = itksys::SystemTools::LowerCase(ext);

    vtkSmartPointer<vtkPNMReader> pnmReader = vtkPNMReader::New();
    pnmReader->SetFileName(this->GetInputLocation().c_str());
    pnmReader->Update();
    vtkSmartPointer<vtkImageData> vtkImageData = pnmReader->GetOutput();

    mitk::Image::Pointer mitkInput = mitk::Image::New();  
    mitkInput->Initialize(vtkImageData);
    mitkInput->SetVolume( vtkImageData->GetScalarPointer() );
    result.push_back(mitkInput.GetPointer());

    setlocale(LC_ALL, currLocale.c_str());
    MITK_DEBUG << "PNM image read";
  }
  catch(...)
  {
    throw;
  }
  return result;
}
