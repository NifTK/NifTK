/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkLabelMapWriter.h"
#include "niftkCoreIOMimeTypes.h"

#include <mitkAbstractFileWriter.h>
#include <mitkCustomMimeType.h>
#include <mitkLogMacros.h>
#include <mitkCommon.h>
#include <vtkSmartPointer.h>
#include <QmitkLookupTableContainer.h>

#include <fstream>

//-----------------------------------------------------------------------------
niftk::LabelMapWriter::LabelMapWriter()
: mitk::AbstractFileWriter(QmitkLookupTableContainer::GetStaticNameOfClass(),
                           mitk::CustomMimeType(niftk::CoreIOMimeTypes::LABELMAP_MIMETYPE_NAME()), 
                           niftk::CoreIOMimeTypes::LABELMAP_MIMETYPE_DESCRIPTION())
{
  RegisterService();
}


//-----------------------------------------------------------------------------
niftk::LabelMapWriter::LabelMapWriter(const niftk::LabelMapWriter & other)
: mitk::AbstractFileWriter(other)
{
}


//-----------------------------------------------------------------------------
niftk::LabelMapWriter::~LabelMapWriter()
{
}

//-----------------------------------------------------------------------------
niftk::LabelMapWriter * niftk::LabelMapWriter::Clone() const
{
  return new niftk::LabelMapWriter(*this);
}


//-----------------------------------------------------------------------------
void niftk::LabelMapWriter::Write()
{

  std::ostream* out;
  std::ofstream outStream;

  if (this->GetOutputStream())
  {
    out = this->GetOutputStream();
  }
  else
  {
    outStream.open(this->GetOutputLocation().c_str());
    out = &outStream;
  }

  if (!out->good())
  {
    MITK_ERROR << "Unable to write to stream.";
  }
  
  std::string outputLocation;
  QmitkLookupTableContainer::ConstPointer lutContainer
    = dynamic_cast<const QmitkLookupTableContainer*>(this->GetInput());

  try
  {
    const std::string& locale = "C";
    const std::string& currLocale = setlocale( LC_ALL, NULL );
    setlocale(LC_ALL, locale.c_str());


    std::locale previousLocale(out->getloc());
    std::locale I("C");
    out->imbue(I);
    
   
    // const_cast here because vtk is stupid and vtkLookupTable->GetTableValue() is not a const function
    vtkLookupTable* unconstTable = const_cast<vtkLookupTable*> (lutContainer->GetLookupTable()); 
    WriteLabelMap(lutContainer->GetLabels(), unconstTable);
    
    setlocale(LC_ALL, currLocale.c_str());
  }
  catch(const std::exception& e)
  {
    MITK_ERROR <<"Exception caught while writing file " <<outputLocation <<": " <<e.what();
    mitkThrow() << e.what();
  }
}


//-----------------------------------------------------------------------------
void niftk::LabelMapWriter::WriteLabelMap(
  LabeledLookupTableProperty::LabelListType labels,
  vtkLookupTable* lookupTable) const
{
  if (labels.empty() || lookupTable == NULL)
  {
    mitkThrow() << "Labels or LookupTable not set.";
  }

  std::ofstream outfile(this->GetOutputLocation().c_str(), std::ofstream::binary);
  
  for (unsigned int i = 0; i < labels.size(); i++)
  {
    int value = labels.at(i).first;  
    QString name = labels.at(i).second;

    // in the slicer file format white space is used to denote space betweeen values, 
    // replacing all white spaces/empty strings with a character to ensure proper IO.
    if (name.isEmpty())
    {
      name = "*";
    }
    else
    {
      name.replace(" ", "*");
    }

    vtkIdType index = lookupTable->GetIndex(value);
    double* rgba = lookupTable->GetTableValue(index);
    int r = rgba[0] * 255;
    int g = rgba[1] * 255;
    int b = rgba[2] * 255;
    int a = rgba[3] * 255;

    std::ostringstream  line;
    outfile << value << " " << name.toStdString() << " "<< r << " " << g << " " << b << " " << a << "\n";
  }

  outfile.flush();
  outfile.close();
}
