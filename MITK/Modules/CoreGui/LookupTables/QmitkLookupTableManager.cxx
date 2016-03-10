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
#include <QFile>
#include <map>
#include <vtkLookupTable.h>
#include <mitkLogMacros.h>
#include <mitkFileReaderRegistry.h>
#include <usGetModuleContext.h>
#include <usModule.h>
#include <usModuleRegistry.h>
#include <mitkLabelMapReader.h>


//-----------------------------------------------------------------------------
QmitkLookupTableManager::QmitkLookupTableManager()
{
  typedef std::pair<int, const QmitkLookupTableContainer*> PairType;
  typedef std::map<int, const QmitkLookupTableContainer*> MapType;

  MapType map;

  // Read all lut files from the resource directory
  QDir fileDir(":");
  fileDir.makeAbsolute();

  QStringList lutFilter;
  lutFilter << "*.lut";
  QStringList lutList = fileDir.entryList(lutFilter, QDir::Files);

  for (int i = 0; i < lutList.size(); i++)
  {
    QString fileName = fileDir.absoluteFilePath(lutList[i]);
    MITK_DEBUG << "QmitkLookupTableManager():Loading lut " << fileName.toLocal8Bit().constData();

    QFile file(fileName);

    QXmlInputSource inputSource(&file);
    QXmlSimpleReader reader;

    QmitkLookupTableSaxHandler handler;
    reader.setContentHandler(&handler);
    reader.setErrorHandler(&handler);

    const QmitkLookupTableContainer *lut = NULL;

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
      map.insert(PairType(lut->GetOrder(), lut));
    }
    else
    {
      MITK_ERROR << "QmitkLookupTableManager():failed to load lookup table:" << fileName.toLocal8Bit().constData();
    }
  }

  QStringList txtFilter;
  txtFilter << "*.txt";
  QStringList labelMapList = fileDir.entryList(txtFilter, QDir::Files,QDir::SortFlag::Name);

  for (int i = 0; i < labelMapList.size(); i++)
  {
    QString fileName = fileDir.filePath(labelMapList[i]);
    MITK_DEBUG << "QmitkLookupTableManager():Loading txt " << fileName.toLocal8Bit().constData();
    
    QFile lutFile(fileName);

    // intialized label map reader
    mitk::LabelMapReader reader;

    reader.SetQFile(lutFile);
    reader.SetOrder(lutList.size() + i);
    reader.Read();

    const QmitkLookupTableContainer *lut = reader.GetLookupTableContainer();
 
    if (lut != NULL)
    {
      map.insert(PairType(lutList.size() + i, lut));
    }
    else
    {
      MITK_ERROR << "QmitkLookupTableManager():failed to load lookup table:" << fileName.toLocal8Bit().constData();
    }
  }

  MapType::iterator iter;
  for (iter = map.begin(); iter != map.end(); iter++)
  {
    m_Containers.emplace((*iter).second->GetDisplayName().toStdString(), (*iter).second);
  }

  MITK_DEBUG << "QmitkLookupTableManager():Constructed, with " << m_Containers.size() << " lookup tables";
}


//-----------------------------------------------------------------------------
QmitkLookupTableManager::~QmitkLookupTableManager()
{
  LookupTableMapType::iterator mapIter;
  for (mapIter = m_Containers.begin(); mapIter != m_Containers.cend(); mapIter++)
  {
    if ((*mapIter).second != NULL)
    {
      delete (*mapIter).second;
    }
  }
  
  m_Containers.clear();
}


//-----------------------------------------------------------------------------
unsigned int QmitkLookupTableManager::GetNumberOfLookupTables()
{
  return m_Containers.size();
}


//-----------------------------------------------------------------------------
const QmitkLookupTableContainer* QmitkLookupTableManager::GetLookupTableContainer(const QString& name)
{
  const QmitkLookupTableContainer* result = NULL;

  if (this->CheckName(name))
  {
    result = m_Containers.at(name.toStdString());
  }
  else
  {
    MITK_ERROR << "GetLookupTableContainer(" << name.toStdString().c_str() << "):invalid name requested, returning NULL";
  }

  return result;
}


//-----------------------------------------------------------------------------
std::vector<QString> QmitkLookupTableManager::GetTableNames()
{
  std::vector<QString> names;

  LookupTableMapType::iterator mapIter;
  for (mapIter = m_Containers.begin(); mapIter != m_Containers.end(); mapIter++)
  {
    names.push_back(QString::fromStdString((*mapIter).first));
  }

  return names;
}


//-----------------------------------------------------------------------------
bool QmitkLookupTableManager::CheckName(const QString& name)
{
  LookupTableMapType::iterator mapIter = m_Containers.find(name.toStdString());
  if (mapIter == m_Containers.end())
  {
    MITK_ERROR << "CheckName(" << name.toStdString().c_str() << ") requested, which does not exist.";
    return false;
  }
  else
  {
    return true;
  }
}


//-----------------------------------------------------------------------------
void QmitkLookupTableManager::AddLookupTableContainer(const QmitkLookupTableContainer *container)
{
  m_Containers.emplace(container->GetDisplayName().toStdString(), container);
}


//-----------------------------------------------------------------------------
void QmitkLookupTableManager::ReplaceLookupTableContainer(const QmitkLookupTableContainer* container, const QString& name)
{
  if (this->CheckName(name))
  {
    m_Containers.at(name.toStdString()) = container;
  }
}
