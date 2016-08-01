/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkLookupTableManager.h"

#include <map>

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QXmlInputSource>
#include <QXmlSimpleReader>

#include <vtkLookupTable.h>

#include <mitkIOUtil.h>
#include <mitkLogMacros.h>
#include <mitkFileReaderRegistry.h>
#include <usGetModuleContext.h>
#include <usModule.h>
#include <usModuleRegistry.h>

#include "niftkLookupTableContainer.h"
#include "niftkLookupTableSaxHandler.h"


namespace niftk
{

//-----------------------------------------------------------------------------
LookupTableManager::LookupTableManager()
{
  typedef std::pair<int, const LookupTableContainer*> PairType;
  typedef std::map<int, const LookupTableContainer*> MapType;

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
    MITK_DEBUG << "LookupTableManager():Loading lut " << fileName.toLocal8Bit().constData();

    QFile file(fileName);

    QXmlInputSource inputSource(&file);
    QXmlSimpleReader reader;

    LookupTableSaxHandler handler;
    reader.setContentHandler(&handler);
    reader.setErrorHandler(&handler);

    const LookupTableContainer *lut = NULL;

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
      map.insert(PairType(lut->GetOrder(), lut));
    }
    else
    {
      MITK_ERROR << "LookupTableManager():failed to load lookup table:" << fileName.toLocal8Bit().constData();
    }
  }

  QStringList txtFilter;
  txtFilter << "*.txt";
  QStringList labelMapList = fileDir.entryList(txtFilter, QDir::Files,QDir::SortFlag::Name);
  for (int i = 0; i < labelMapList.size(); i++)
  {
    LookupTableContainer *lut = NULL;
    QString fileName = fileDir.filePath(labelMapList[i]);
    MITK_DEBUG << "LookupTableManager():Loading txt " << fileName.toLocal8Bit().constData();

    mitk::FileReaderRegistry* frr = new mitk::FileReaderRegistry();

    mitk::MimeType myMimeType = frr->GetMimeTypeForFile(fileName.toStdString());
    std::vector<mitk::FileReaderRegistry::ReaderReference> refs = frr->GetReferences(myMimeType);

    if (refs.empty())
    {
      MITK_ERROR << "No references found for mime type: " << myMimeType.GetName();
    }

    mitk::IFileReader* myReader = frr->GetReader(refs.at(0));

    if (myReader == NULL)
    {
      MITK_ERROR << "No reader found for mime type: " << myMimeType.GetName();
    }

    myReader->SetInput(fileName.toStdString());
    std::vector<mitk::BaseData::Pointer> container = myReader->Read(); // this will never work because the file does not exist!¬!!!!
    if (container.empty())
    {
      MITK_ERROR << "Unable to load LookupTableContainer from " << fileName.toStdString();
    }
    else
    {
      lut =
        dynamic_cast<LookupTableContainer* >(container.at(0).GetPointer());

      if (lut != NULL)
      {
        lut->SetOrder(lutList.size() + i);
        map.insert(PairType(lutList.size() + i, lut));
      }
      else
      {
        MITK_ERROR << "LookupTableManager():failed to load lookup table:" << fileName.toLocal8Bit().constData();
      }
    }
  }

  MapType::iterator iter;
  for (iter = map.begin(); iter != map.end(); iter++)
  {
    m_Containers.emplace((*iter).second->GetDisplayName().toStdString(), (*iter).second);
  }

  MITK_DEBUG << "LookupTableManager():Constructed, with " << m_Containers.size() << " lookup tables";
}


//-----------------------------------------------------------------------------
LookupTableManager::~LookupTableManager()
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
unsigned int LookupTableManager::GetNumberOfLookupTables()
{
  return m_Containers.size();
}


//-----------------------------------------------------------------------------
const LookupTableContainer* LookupTableManager::GetLookupTableContainer(const QString& name)
{
  const LookupTableContainer* result = NULL;

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
std::vector<QString> LookupTableManager::GetTableNames()
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
bool LookupTableManager::CheckName(const QString& name)
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
void LookupTableManager::AddLookupTableContainer(const LookupTableContainer *container)
{
  m_Containers.emplace(container->GetDisplayName().toStdString(), container);
}


//-----------------------------------------------------------------------------
void LookupTableManager::ReplaceLookupTableContainer(const LookupTableContainer* container, const QString& name)
{
  if (this->CheckName(name))
  {
    m_Containers.at(name.toStdString()) = container;
  }
}

}
