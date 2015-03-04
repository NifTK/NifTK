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

#include "mitkPNMWriter.h"
#include "mitkPNMIOMimeTypes.h"

#include <vtkSmartPointer.h>
#include <vtkCleanPolyData.h>
#include <vtkPNMReader.h>
#include <vtkPNMWriter.h>
#include <vtkImageData.h>

#include <itksys/SystemTools.hxx>
#include <itkSize.h>

#include <mitkAbstractFileWriter.h>
#include <mitkCustomMimeType.h>
#include <mitkLogMacros.h>
#include <mitkImage.h>



mitk::PNMWriter::PNMWriter()
  : mitk::AbstractFileWriter("PNMWriter", CustomMimeType( mitk::PNMIOMimeTypes::PNM_MIMETYPE_NAME() ), mitk::PNMIOMimeTypes::PNM_MIMETYPE_DESCRIPTION())
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

  try
  {
    const std::string& locale = "C";
    const std::string& currLocale = setlocale( LC_ALL, NULL );
    setlocale(LC_ALL, locale.c_str());


    std::locale previousLocale(out->getloc());
    std::locale I("C");
    out->imbue(I);

    std::string filename = this->GetOutputLocation().c_str();

    mitk::Image::ConstPointer input = dynamic_cast<const mitk::Image*>(this->GetInput());
    std::string ext = itksys::SystemTools::GetFilenameLastExtension(this->GetOutputLocation().c_str());

    // default extension is .fib
    if(ext == "")
    {
      ext = ".ppm";
      this->SetOutputLocation(this->GetOutputLocation() + ext);
    }

    if (ext==".pbm" || ext==".PBM")
    {
      itksys::SystemTools::ReplaceString(filename,".ppm",".pbm");
      MITK_INFO << "Writing image as Portable BitMap (PBM)";
    }
    else if (ext==".pgm" || ext==".PGM")
    {
      itksys::SystemTools::ReplaceString(filename,".ppm",".pbm");
      MITK_INFO << "Writing image as Portable GreyMap (PGM)";
    }
    else if (ext==".ppm" || ext==".PPM")
    {
      MITK_INFO << "Writing image as Portable PixelMap (PPM)";
    }

    vtkSmartPointer<vtkPNMWriter> pnmWriter = vtkPNMWriter::New();
    pnmWriter->SetFileName((this->GetOutputLocation().c_str()));
    vtkImageData * nonConstImg = const_cast<vtkImageData *>(input->GetVtkImageData());
    pnmWriter->SetInputData(nonConstImg);
    pnmWriter->Write();

    setlocale(LC_ALL, currLocale.c_str());
    MITK_INFO << "PNM image written";
  }
  catch(...)
  {
    throw;
  }
}
