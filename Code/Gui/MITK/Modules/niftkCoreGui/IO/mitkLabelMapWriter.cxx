/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkLabelMapWriter.h"
#include "../Internal/niftkCoreGuiIOMimeTypes.h"

#include <mitkAbstractFileWriter.h>
#include <mitkCustomMimeType.h>
#include <mitkLogMacros.h>
#include <mitkCommon.h>
#include <vtkSmartPointer.h>

#include <fstream>


mitk::LabelMapWriter::LabelMapWriter()
  : mitk::AbstractFileWriter("Label Map", CustomMimeType(niftk::CoreGuiIOMimeTypes::LABELMAP_MIMETYPE_NAME() ), niftk::CoreGuiIOMimeTypes::LABELMAP_MIMETYPE_DESCRIPTION())
{
  RegisterService();
}

mitk::LabelMapWriter::LabelMapWriter(const mitk::LabelMapWriter & other)
  :mitk::AbstractFileWriter(other)
{}

mitk::LabelMapWriter * mitk::LabelMapWriter::Clone() const
{
  return new mitk::LabelMapWriter(*this);
}

void mitk::LabelMapWriter::Write()
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
    MITK_ERROR << "Unable to write to stream.";
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
    
    WriteLabelMap();
    
    setlocale(LC_ALL, currLocale.c_str());
  }
  catch(const std::exception& e)
  {
    MITK_ERROR <<"Exception caught while writing file " <<outputLocation <<": " <<e.what();
    mitkThrow() << e.what();
  }
}

void mitk::LabelMapWriter::WriteLabelMap()
{
  if(m_Labels.empty() || m_LookupTable == NULL)
  {
    mitkThrow() << "Labels or LookupTable not set.";
  }

  std::ofstream outfile( this->GetOutputLocation().c_str(), std::ofstream::binary);
  
  for( unsigned int i=0; i<m_Labels.size();i++ )
  {
    int value = m_Labels.at(i).first; 
    
    QString name = m_Labels.at(i).second;

    // in the slicer file format white space is used to denote space betweeen values, 
    // replacing all white spaces/empty strings with a character to ensure proper IO.
    if( name.isEmpty() )
      name = "*";
    else
      name.replace(" ", "*");

    vtkIdType index = m_LookupTable->GetIndex(value);
    double* rgba = m_LookupTable->GetTableValue(value);
    int r = rgba[0]*255;
    int g = rgba[1]*255;
    int b = rgba[2]*255;
    int a = rgba[3]*255;

    std::ostringstream  line;
    outfile << value << " " << name.toStdString() << " "<< r << " " << g << " " << b << " " << a << "\n";
  }

  outfile.flush();
  outfile.close();
}
