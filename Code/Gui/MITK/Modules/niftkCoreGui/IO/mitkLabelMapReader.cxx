/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkLabelMapReader.h"
#include "../Internal/niftkCoreGuiIOMimeTypes.h"

#include <mitkCustomMimeType.h>
#include <mitkLogMacros.h>

#include <itksys/SystemTools.hxx>
#include <sstream>

mitk::LabelMapReader::LabelMapReader()
  : mitk::AbstractFileReader(CustomMimeType( niftk::CoreGuiIOMimeTypes::LABELMAP_MIMETYPE_NAME() ), niftk::CoreGuiIOMimeTypes::LABELMAP_MIMETYPE_DESCRIPTION() )
{
  m_ServiceReg = this->RegisterService();
}

mitk::LabelMapReader::LabelMapReader(const LabelMapReader &other)
  :mitk::AbstractFileReader(other)
{
}

mitk::LabelMapReader * mitk::LabelMapReader::Clone() const
{
  return new mitk::LabelMapReader(*this);
}


std::vector<itk::SmartPointer<mitk::BaseData> > mitk::LabelMapReader::Read()
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

    this->ReadLabelMap();
    
    setlocale(LC_ALL, currLocale.c_str());
    MITK_DEBUG << "NifTK label map readed";
  }
  catch(...)
  {
    throw;
  }
  return result;
}

void mitk::LabelMapReader::ReadLabelMap()
{
  std::string inputRaw;
  QString descriptorString;
  descriptorString.clear();

  QString fName;
  fName.append(this->GetInputLocation().c_str());

  std::ifstream infile(fName.toStdString().c_str(), std::ifstream::in);
  if (infile.is_open())
  {
    //while there are still lines in the file, keep reading:
    while (!infile.eof())
    {
      std::string line;
      //place the line from input into the raw string
      getline(infile, line);

      if( line.empty() || line.at(0) == '#')
        continue;

      try
      {
        int value,red,green,blue,alpha;
      
        // find value
        size_t firstSpace = line.find_first_of(' ');
        std::string firstDigit = line.substr(0, firstSpace);
        sscanf(firstDigit.c_str(), "%i", &value);

        // find name
        size_t firstLtr = line.find_first_not_of(' ', firstSpace);
        size_t lastLtr  = line.find_first_of(' ', firstLtr);

        std::string nameInFile = line.substr(firstLtr, (lastLtr)-firstLtr);
        QString name = QString::fromStdString(nameInFile);

        // if the name is just the special character set as empty
        if(name.compare(QString('*'))==0)
          name.clear();

        name.replace('*',' '); // swapping the white space back in

        // colors;
        std::string colorStr = line.substr(lastLtr, line.size() - lastLtr);
        sscanf(colorStr.c_str(), "%i %i %i %i", &red, &green, &blue, &alpha);

	      LabelMapItem label;
        label.value = value;
        label.name  = name;
	  
  	    QColor fileColor(red, green, blue, alpha);
	      label.color = fileColor;

  	    m_Labels.push_back(label);
      }
      catch(...)
      {
        std::cout <<"Unable to parse line " << line.c_str() << ". Skipping." << std::endl;
      }
    }
  }
  infile.close();

}