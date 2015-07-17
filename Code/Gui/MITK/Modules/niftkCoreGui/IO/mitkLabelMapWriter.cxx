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
  std::ofstream outfile( this->GetOutputLocation().c_str(), std::ofstream::binary);
  
  for( unsigned int i=0; i<m_Labels.size();i++ )
  {
    int value = m_Labels.at(i).value; 
    
    QString name = m_Labels.at(i).name;

    // in the slicer file format white space is used to denote space betweeen values, 
    // replacing all white spaces/empty strings with a character to ensure proper IO.
    if( name.isEmpty() )
      name = "*";
    else
      name.replace(" ", "*");

    int red = m_Labels.at(i).color.red();
    int green = m_Labels.at(i).color.green();
    int blue = m_Labels.at(i).color.blue();
    int alpha = m_Labels.at(i).color.alpha();

    std::ostringstream  line;
    outfile << value << " " << name.toStdString() << " "<< red << " " << green << " " << blue << " " << alpha << "\n";
  }

  outfile.flush();
  outfile.close();
}
