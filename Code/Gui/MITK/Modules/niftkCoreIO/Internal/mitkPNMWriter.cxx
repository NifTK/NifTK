/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkPNMWriter.h"
#include "../Internal/niftkCoreIOMimeTypes.h"

#include <vtkSmartPointer.h>
#include <vtkCleanPolyData.h>
#include <vtkPNMReader.h>
#include <vtkPNMWriter.h>
#include <vtkImageData.h>

#include <mitkAbstractFileWriter.h>
#include <mitkCustomMimeType.h>
#include <mitkLogMacros.h>
#include <mitkImage.h>



mitk::PNMWriter::PNMWriter()
  : mitk::AbstractFileWriter(mitk::Image::GetStaticNameOfClass(), CustomMimeType(niftk::CoreIOMimeTypes::PNM_MIMETYPE_NAME() ), niftk::CoreIOMimeTypes::PNM_MIMETYPE_DESCRIPTION())
{
  RegisterService();
}

mitk::PNMWriter::PNMWriter(const mitk::PNMWriter & other)
  :mitk::AbstractFileWriter(other)
{}

mitk::PNMWriter::~PNMWriter()
{}

mitk::PNMWriter * mitk::PNMWriter::Clone() const
{
  return new mitk::PNMWriter(*this);
}

void mitk::PNMWriter::Write()
{

  std::ostream* out;
  std::ofstream outStream;

  if( this->GetOutputStream() )
  {
    out = this->GetOutputStream();
  }
  else
  {
    outStream.open( this->GetOutputLocation().c_str() );
    out = &outStream;
  }

  if ( !out->good() )
  {
    MITK_ERROR << "Stream not good.";
  }
  
  std::string outputLocation;

  try
  {
    const std::string& locale = "C";
    const std::string& currLocale = setlocale( LC_ALL, NULL );
    setlocale(LC_ALL, locale.c_str());


    std::locale previousLocale(out->getloc());
    std::locale I("C");
    out->imbue(I);

    mitk::Image::ConstPointer input = dynamic_cast<const mitk::Image*>(this->GetInput());

    std::string outputLocation = this->GetOutputLocation().c_str();
    vtkSmartPointer<vtkPNMWriter> pnmWriter = vtkPNMWriter::New();
    pnmWriter->SetFileName(outputLocation.c_str());
    vtkImageData * nonConstImg = const_cast<vtkImageData *>(input->GetVtkImageData());
    pnmWriter->SetInputData(nonConstImg);
    pnmWriter->Write();

    setlocale(LC_ALL, currLocale.c_str());
  }
  catch(const std::exception& e)
  {
    MITK_ERROR <<"Exception caught while writing file " <<outputLocation <<": " <<e.what();
    mitkThrow() << e.what();
  }
}
