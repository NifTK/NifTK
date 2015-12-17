/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
#include "niftkIGIDataSourceUtils.h"

namespace niftk
{

//-----------------------------------------------------------------------------
QString GetPreferredSlash()
{
  return QString(QDir::separator());
}


//-----------------------------------------------------------------------------
std::set<niftk::IGIDataType::IGITimeType> ProbeTimeStampFiles(QDir path, const QString& suffix)
{
  static_assert(std::numeric_limits<qulonglong>::max() >= std::numeric_limits<niftk::IGIDataType::IGITimeType>::max(), "mismatched data types");

  std::set<niftk::IGIDataType::IGITimeType>  result;

  QStringList filters;
  filters << QString("*" + suffix);
  path.setNameFilters(filters);
  path.setFilter(QDir::Files | QDir::Readable | QDir::NoDotAndDotDot);

  QStringList files = path.entryList();
  if (!files.empty())
  {
    foreach (QString file, files)
    {
      if (file.endsWith(suffix))
      {
        // in the future, maybe add prefix parsing too.
        QString  middle = file.mid(0, file.size() - suffix.size());

        bool  ok = false;
        qulonglong value = middle.toULongLong(&ok);
        if (ok)
        {
          result.insert(value);
        }
      }
    }
  }

  return result;
}


//-----------------------------------------------------------------------------
QMap<QString, std::set<niftk::IGIDataType::IGITimeType> >
GetPlaybackIndex(const QString& directory, const QString& fileExtension)
{
  QMap<QString, std::set<niftk::IGIDataType::IGITimeType> > result;

  QDir recordingDir(directory);
  if (recordingDir.exists())
  {
    // then directories with tool/source names
    recordingDir.setFilter(QDir::Dirs | QDir::Readable | QDir::NoDotAndDotDot);

    QStringList toolNames = recordingDir.entryList();
    if (!toolNames.isEmpty())
    {
      foreach (QString tool, toolNames)
      {
        QDir  tooldir(recordingDir.path() + QDir::separator() + tool);
        assert(tooldir.exists());

        std::set<niftk::IGIDataType::IGITimeType> timeStamps = ProbeTimeStampFiles(tooldir, fileExtension);
        if (!timeStamps.empty())
        {
          result.insert(tool, timeStamps);
        }
      }
    }
    else
    {
      MITK_WARN << "There are no tool sub-folders in " << recordingDir.absolutePath().toStdString() << ", so can't playback data!";
      return result;
    }
  }
  else
  {
    mitkThrow() << "Recording directory, " << recordingDir.absolutePath().toStdString() << ", does not exist!";
  }
  if (result.isEmpty())
  {
    mitkThrow() << "No data extracted from directory " << recordingDir.absolutePath().toStdString();
  }
  return result;
}


//-----------------------------------------------------------------------------
bool ProbeRecordedData(const QString& path,
                       const QString& fileExtension,
                       niftk::IGIDataType::IGITimeType* firstTimeStampInStore,
                       niftk::IGIDataType::IGITimeType* lastTimeStampInStore)
{

  niftk::IGIDataType::IGITimeType  firstTimeStampFound = std::numeric_limits<niftk::IGIDataType::IGITimeType>::max();
  niftk::IGIDataType::IGITimeType  lastTimeStampFound  = std::numeric_limits<niftk::IGIDataType::IGITimeType>::min();

  // Note, that each tool may have different min and max, so we want the
  // most minimum and most maximum of all the sub directories.

  QMap<QString, std::set<niftk::IGIDataType::IGITimeType> > result = niftk::GetPlaybackIndex(path, fileExtension);
  if (result.isEmpty())
  {
    return false;
  }

  QMap<QString, std::set<niftk::IGIDataType::IGITimeType> >::iterator iter;
  for (iter = result.begin(); iter != result.end(); iter++)
  {
    if (!iter.value().empty())
    {
      niftk::IGIDataType::IGITimeType first = *((*iter).begin());
      if (first < firstTimeStampFound)
      {
        firstTimeStampFound = first;
      }

      niftk::IGIDataType::IGITimeType last = *(--((*iter).end()));
      if (last > lastTimeStampFound)
      {
        lastTimeStampFound = last;
      }
    }
  }
  if (firstTimeStampInStore)
  {
    *firstTimeStampInStore = firstTimeStampFound;
  }
  if (lastTimeStampInStore)
  {
    *lastTimeStampInStore = lastTimeStampFound;
  }
  return firstTimeStampFound != std::numeric_limits<niftk::IGIDataType::IGITimeType>::max();
}

} // end namespace

