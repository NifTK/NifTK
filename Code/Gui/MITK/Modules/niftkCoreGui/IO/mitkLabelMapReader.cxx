/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkLabelMapReader.h"
#include "niftkCoreGuiIOMimeTypes.h"
#include "QmitkLookupTableContainer.h"

#include <mitkCustomMimeType.h>
#include <mitkLogMacros.h>

#include <vtkSmartPointer.h>
#include <vtkLookupTable.h>

#include <sstream>
#include <iostream>

//-----------------------------------------------------------------------------
mitk::LabelMapReader::LabelMapReader()
  : mitk::AbstractFileReader(CustomMimeType( niftk::CoreGuiIOMimeTypes::LABELMAP_MIMETYPE_NAME() ), niftk::CoreGuiIOMimeTypes::LABELMAP_MIMETYPE_DESCRIPTION() )
{
  m_ServiceReg = this->RegisterService();
}


//-----------------------------------------------------------------------------
mitk::LabelMapReader::LabelMapReader(const LabelMapReader &other)
  :mitk::AbstractFileReader(other)
{
}


//-----------------------------------------------------------------------------
mitk::LabelMapReader * mitk::LabelMapReader::Clone() const
{
  return new mitk::LabelMapReader(*this);
}


//-----------------------------------------------------------------------------
std::vector<itk::SmartPointer<mitk::BaseData> > mitk::LabelMapReader::Read()
{

  std::vector<itk::SmartPointer<mitk::BaseData> > result;
  try
  {
    const std::string& locale = "C";
    const std::string& currLocale = setlocale( LC_ALL, NULL );
    setlocale(LC_ALL, locale.c_str());

    std::string fileName = this->GetInputLocation();
    std::ifstream infile(fileName, std::ifstream::in);

    bool isLoaded = false;
    QString labelName;
    if( infile.is_open() )
    {
      labelName = QString::fromStdString(fileName);
      isLoaded = this->ReadLabelMap(infile);
      infile.close();
    }
    else
    {
      m_InputQFile->open(QIODevice::ReadOnly);   
      labelName = m_InputQFile->fileName();

      // this is a dirt hack to get the resource file in the right format to read
      QDataStream qstream(m_InputQFile);

      std::string fileStr(m_InputQFile->readAll());
      std::stringstream sStream; 
      sStream << fileStr;

      isLoaded = this->ReadLabelMap(sStream);
    }

    if( isLoaded )
    {
      int startInd = labelName.lastIndexOf("/")+1;
      int endInd = labelName.lastIndexOf(".");
      m_DisplayName = labelName.mid(startInd, endInd-startInd);
      setlocale(LC_ALL, currLocale.c_str());
      MITK_DEBUG << "NifTK label map readed";
    }
    else
    {
      result.clear();
      MITK_ERROR << "Unable to read NifTK label map!";
    }
  }
  catch(...)
  {
    throw;
  }
  return result;
}


//-----------------------------------------------------------------------------
bool mitk::LabelMapReader::ReadLabelMap(std::istream & file)
{
  bool isLoaded = false;

  //while there are still lines in the file, keep reading:
  while (!file.eof())
  {
    std::string line;
    //place the line from input into the raw string
    getline(file, line);

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
      isLoaded = true;
    }
    catch(...)
    {
      std::cout <<"Unable to parse line " << line.c_str() << ". Skipping." << std::endl;
    }
  }

  return isLoaded;
}


//-----------------------------------------------------------------------------
QmitkLookupTableContainer* mitk::LabelMapReader::GetLookupTableContainer()
{
  if(m_Labels.size() < 1)
    return NULL;

  MITK_DEBUG << "GetLookupTableContainer():labels.size()=" << m_Labels.size();

  // get the size of vtkLUT from the range of values
  int min = m_Labels.at(0).value;
  int max = min;

  for(unsigned int i=1;i<m_Labels.size();i++)
  {
    int val = m_Labels.at(i).value;
    if(val<min)
      min = val;
    else if (val>max)
      max = val;
  }

  vtkSmartPointer<vtkLookupTable> lookupTable = vtkLookupTable::New();
  
  //because vtk is stupid to initialize with empty settings I have to restrict all ranges to 0
  lookupTable->SetValueRange(0,0);
  lookupTable->SetHueRange(0,0);
  lookupTable->SetSaturationRange(0,0);
  lookupTable->SetAlphaRange(0,0);

  // number of table values
  int numberOfValues = (max-min)+2;
  lookupTable->SetNumberOfTableValues( numberOfValues ); 
  lookupTable->SetTableRange(min-1,max+1);
  lookupTable->SetNanColor(0,0,0,0);

  //vtkLUT->SetIndexedLookup(true);  
  lookupTable->Build();

  for( unsigned int i=0;i<m_Labels.size();i++)
  {
    double value = m_Labels.at(i).value - min+1;
    double r = double(m_Labels.at(i).color.red())/255;
    double g = double(m_Labels.at(i).color.green())/255;
    double b = double(m_Labels.at(i).color.blue())/255;
    double a = double(m_Labels.at(i).color.alpha())/255;
    
    lookupTable->SetTableValue(value,r,g,b,a);
  }

  QmitkLookupTableContainer *lookupTableContainer = new QmitkLookupTableContainer(lookupTable);
  lookupTableContainer->SetIsScaled(false);
  lookupTableContainer->SetDisplayName(m_DisplayName);
  lookupTableContainer->SetOrder(m_Order);

  return lookupTableContainer;
}
