/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkLookupTableManager.h"
#include "QmitkLookupTableContainer.h"
#include "QmitkLookupTableSaxHandler.h"
#include <QXmlInputSource>
#include <QXmlSimpleReader>
#include <QCoreApplication>
#include <QDir>
#include <map>
#include <vtkLookupTable.h>
#include <mitkLogMacros.h>

//-----------------------------------------------------------------------------
QmitkLookupTableManager::QmitkLookupTableManager()
{
  typedef std::pair<int, const QmitkLookupTableContainer*> PairType;
  typedef std::map<int, const QmitkLookupTableContainer*> MapType;

	MapType map;

	// TODO: How can I automatically read a list of filenames within a plugin?
  QDir directory(":");
  QStringList filters;
  filters << "*.lut";
	QStringList fileList = directory.entryList(filters, QDir::Files);

	for (int i = 0; i < fileList.size(); i++)
	{
    QString fileName = directory.absoluteFilePath(fileList[i]);
    MITK_DEBUG << "QmitkLookupTableManager():Loading lut " << fileName.toLocal8Bit().constData();

    QFile file(fileName);

	  QXmlInputSource inputSource(&file);
	  QXmlSimpleReader reader;

    QmitkLookupTableSaxHandler handler;
	  reader.setContentHandler(&handler);
	  reader.setErrorHandler(&handler);

    QmitkLookupTableContainer *lut = NULL;

	  if (reader.parse(inputSource))
	  {
	  	lut = handler.GetLookupTableContainer();
	  }
	  else
	  {
      MITK_ERROR << "QmitkLookupTableManager():failed to parse XML file (" << fileName.toLocal8Bit().constData() \
	  			<< ") so returning null";
	  }

		if (lut != NULL)
		{
      map.insert(PairType(lut->GetOrder(), const_cast<const QmitkLookupTableContainer*>(lut)));
		}
		else
		{
      MITK_ERROR << "QmitkLookupTableManager():failed to load lookup table:" << fileName.toLocal8Bit().constData();
		}
	}

	MapType::iterator iter;
	for (iter = map.begin(); iter != map.end(); iter++)
	{
		m_List.push_back((*iter).second);
	}

  MITK_DEBUG << "QmitkLookupTableManager():Constructed, with " << m_List.size() << " lookup tables";
}


//-----------------------------------------------------------------------------
QmitkLookupTableManager::~QmitkLookupTableManager()
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


//-----------------------------------------------------------------------------
unsigned int QmitkLookupTableManager::GetNumberOfLookupTables()
{
	return m_List.size();
}


//-----------------------------------------------------------------------------
const QmitkLookupTableContainer* QmitkLookupTableManager::GetLookupTableContainer(const unsigned int& n)
{
  const QmitkLookupTableContainer* result = NULL;

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


//-----------------------------------------------------------------------------
QString QmitkLookupTableManager::GetName(const unsigned int& n)
{
  QString result = "";
  const QmitkLookupTableContainer* container = this->GetLookupTableContainer(n);
  if (container != 0)
  {
    result = container->GetDisplayName();
  }
  return result;
}


//-----------------------------------------------------------------------------
bool QmitkLookupTableManager::CheckIndex(const unsigned int& n)
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
