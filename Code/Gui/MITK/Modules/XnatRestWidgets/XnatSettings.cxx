/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "XnatSettings.h"

#include <QDir>
#include <QFileInfo>
#include <QUuid>

#include "XnatLoginProfile.h"

XnatSettings::XnatSettings()
{
}

XnatSettings::~XnatSettings()
{
}

QString XnatSettings::getWorkSubdirectory() const
{
  // set work directory name
  QDir workDir;
  QString workDirName = getDefaultWorkDirectory();
  if ( !workDirName.isEmpty() )
  {
    workDir = QDir(workDirName);
  }

  // generate random name for subdirectory
  QString subdir = QUuid::createUuid().toString();

  // create subdirectory in work directory
  bool subdirCreated = workDir.mkdir(subdir);

  // check whether subdirectory was created
  if ( !subdirCreated )
  {
    // display error message
    return QString();
  }

  // return full path of subdirectory
  return QFileInfo(workDir, subdir).absoluteFilePath();
}
