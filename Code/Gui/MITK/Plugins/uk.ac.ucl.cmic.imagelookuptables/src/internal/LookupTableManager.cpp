/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-04-08 14:15:23 +0100 (Fri, 08 Apr 2011) $
 Revision          : $Revision: 5819 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef LOOKUPTABLEMANAGER_CPP
#define LOOKUPTABLEMANAGER_CPP
#include <QXmlInputSource>
#include <QXmlSimpleReader>
#include <QCoreApplication>
#include <QDir>
#include <map>
#include "LookupTableManager.h"
#include "LookupTableSaxHandler.h"
#include "LookupTableContainer.h"
#include "vtkLookupTable.h"
#include "mitkLogMacros.h"

LookupTableManager::LookupTableManager()
{
	typedef std::pair<int, const LookupTableContainer*> PairType;
	typedef std::map<int, const LookupTableContainer*> MapType;

	MapType map;

	// TODO: How can I automatically read a list of filenames within a plugin?
	QStringList fileList;
  fileList.push_back(":/ImageLookupTables/imagej_fire.lut");
	fileList.push_back(":/ImageLookupTables/blue.lut");
  fileList.push_back(":/ImageLookupTables/cyan.lut");
  fileList.push_back(":/ImageLookupTables/green.lut");
  fileList.push_back(":/ImageLookupTables/grey.lut");
  fileList.push_back(":/ImageLookupTables/hot.lut");
  fileList.push_back(":/ImageLookupTables/hsv.lut");
  fileList.push_back(":/ImageLookupTables/jet.lut");
  fileList.push_back(":/ImageLookupTables/magenta.lut");
  fileList.push_back(":/ImageLookupTables/matlab_autumn.lut");
  fileList.push_back(":/ImageLookupTables/matlab_bipolar_256_0.1.lut");
  fileList.push_back(":/ImageLookupTables/matlab_bipolar_256_0.9.lut");
  fileList.push_back(":/ImageLookupTables/matlab_cool.lut");
  fileList.push_back(":/ImageLookupTables/matlab_hot.lut");
  fileList.push_back(":/ImageLookupTables/matlab_spring.lut");
  fileList.push_back(":/ImageLookupTables/matlab_summer.lut");
  fileList.push_back(":/ImageLookupTables/matlab_winter.lut");
  fileList.push_back(":/ImageLookupTables/midas_bands.lut");
  fileList.push_back(":/ImageLookupTables/midas_hot_iron.lut");
  fileList.push_back(":/ImageLookupTables/midas_overlay.lut");
  fileList.push_back(":/ImageLookupTables/midas_pet_map.lut");
  fileList.push_back(":/ImageLookupTables/midas_spectrum.lut");
  fileList.push_back(":/ImageLookupTables/nih.lut");
  fileList.push_back(":/ImageLookupTables/red.lut");
  fileList.push_back(":/ImageLookupTables/sea.lut");
  fileList.push_back(":/ImageLookupTables/yellow.lut");

	for (int i = 0; i < fileList.size(); i++)
	{

	  QString fileName = fileList[i];
	  MITK_DEBUG << "LookupTableManager():Loading lut " << fileName.toLocal8Bit().constData();

    QFile file(fileName);

	  QXmlInputSource inputSource(&file);
	  QXmlSimpleReader reader;

    LookupTableSaxHandler handler;
	  reader.setContentHandler(&handler);
	  reader.setErrorHandler(&handler);

		LookupTableContainer *lut = NULL;

	  if (reader.parse(inputSource))
	  {
	  	lut = handler.GetLookupTableContainer();
	  }
	  else
	  {
	    MITK_ERROR << "LookupTableManager():failed to parse XML file (" << fileName.toLocal8Bit().constData() \
	  			<< ") so returning null";
	  }

		if (lut != NULL)
		{
			map.insert(PairType(lut->GetOrder(), const_cast<const LookupTableContainer*>(lut)));
		}
		else
		{
		  MITK_ERROR << "LookupTableManager():failed to load lookup table:" << fileName.toLocal8Bit().constData();
		}
	}

	MapType::iterator iter;
	for (iter = map.begin(); iter != map.end(); iter++)
	{
		m_List.push_back((*iter).second);
	}

	MITK_DEBUG << "LookupTableManager():Constructed, with " << m_List.size() << " lookup tables";
}

LookupTableManager::~LookupTableManager()
{
	for (unsigned int i = 0; i < m_List.size(); i++)
	{
	  if (m_List[i] != NULL)
	  {
	    delete m_List[i];
	  }
	}
	m_List.clear();
}

unsigned int LookupTableManager::GetNumberOfLookupTables()
{
	return m_List.size();;
}

const LookupTableContainer* LookupTableManager::GetLookupTableContainer(const unsigned int& n)
{
	const LookupTableContainer* result = NULL;

	if (this->CheckIndex(n))
	{
		result = m_List[n];
	}
	else
	{
	  MITK_ERROR << "GetLookupTableContainer(" << n << "):invalid index requested, returning NULL";
	}

	return result;
}

vtkLookupTable* LookupTableManager::CloneLookupTable(const unsigned int& n)
{
  vtkLookupTable *result = NULL;

  const LookupTableContainer* container = NULL;
  container = this->GetLookupTableContainer(n);

  if (container != NULL)
  {
    result->DeepCopy(const_cast<vtkLookupTable*>(container->GetLookupTable()));
  }

  return result;
}

bool LookupTableManager::CheckIndex(const unsigned int& n)
{
	if (n >= this->GetNumberOfLookupTables())
	{
	  MITK_ERROR << "CheckIndex(" << n << ") requested, which is out of range";
		return false;
	}
	else
	{
		return true;
	}
}

#endif
