/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkIGIDataSource.h"
#include "QmitkIGIDataSourceBackgroundSaveThread.h"

//-----------------------------------------------------------------------------
QmitkIGIDataSource::QmitkIGIDataSource(mitk::DataStorage* storage)
: mitk::IGIDataSource(storage)
, m_SaveThread(NULL)
{
  m_SaveThread = new QmitkIGIDataSourceBackgroundSaveThread(this, this);
}


//-----------------------------------------------------------------------------
QmitkIGIDataSource::~QmitkIGIDataSource()
{
  if (m_SaveThread != NULL)
  {
    m_SaveThread->ForciblyStop();
    delete m_SaveThread;
  }
}


//-----------------------------------------------------------------------------
void QmitkIGIDataSource::EmitDataSourceStatusUpdatedSignal()
{
  emit DataSourceStatusUpdated(this->GetIdentifier());
}


//-----------------------------------------------------------------------------
void QmitkIGIDataSource::SetSavingMessages(bool isSaving)
{
  // FIXME: race-condition between data-grabbing thread and UI thread setting m_SavingMessages!
  mitk::IGIDataSource::SetSavingMessages(isSaving);
  if (!m_SaveThread->isRunning())
  {
    m_SaveThread->start();
  }
}


//-----------------------------------------------------------------------------
void QmitkIGIDataSource::SetSavingInterval(int seconds)
{
  m_SaveThread->SetInterval(seconds*1000);
  this->Modified();
}


//-----------------------------------------------------------------------------
std::set<igtlUint64> QmitkIGIDataSource::ProbeTimeStampFiles(QDir path, const QString& extension)
{
  // this should be a static assert...
  assert(std::numeric_limits<qulonglong>::max() >= std::numeric_limits<igtlUint64>::max());

  std::set<igtlUint64>  result;

  QStringList filters;
  filters << QString("*." + extension);
  path.setNameFilters(filters);
  path.setFilter(QDir::Files | QDir::Readable | QDir::NoDotAndDotDot);

  QStringList files = path.entryList();
  if (!files.empty())
  {
    foreach (QString file, files)
    {
      //std::cout << file.toStdString() << std::endl;

      QStringList parts = file.split('.');
      if (parts.size() == 2)
      {
        if (parts[1] == extension)
        {
          bool  ok = false;
          qulonglong value = parts[0].toULongLong(&ok);
          if (ok)
          {
            result.insert(value);
          }
        }
      }
    }
  }

  return result;
}

