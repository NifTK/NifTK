/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "StereoCameraCalibrationSelectionWidget.h"
#include <QDir>

//-----------------------------------------------------------------------------
StereoCameraCalibrationSelectionWidget::StereoCameraCalibrationSelectionWidget(QWidget *parent)
{
  setupUi(this);
  connect(m_LeftIntrinsicEdit, SIGNAL(currentPathChanged(QString)), this, SLOT(OnLeftIntrinsicCurrentPathChanged(QString)));
  connect(m_RightIntrinsicEdit, SIGNAL(currentPathChanged(QString)), this, SLOT(OnRightIntrinsicCurrentPathChanged(QString)));
  connect(m_LeftRightTransformEdit, SIGNAL(currentPathChanged(QString)), this, SLOT(OnLeftToRightTransformChanged(QString)));
}


//-----------------------------------------------------------------------------
QString StereoCameraCalibrationSelectionWidget::GetLastDirectory() const
{
  return m_LastDirectory;
}


//-----------------------------------------------------------------------------
void StereoCameraCalibrationSelectionWidget::SetLastDirectory(const QString& dirName)
{
  m_LastDirectory = dirName;
}


//-----------------------------------------------------------------------------
QString StereoCameraCalibrationSelectionWidget::GetLeftIntrinsicFileName() const
{
  return m_LeftIntrinsicEdit->currentPath();
}


//-----------------------------------------------------------------------------
QString StereoCameraCalibrationSelectionWidget::GetRightIntrinsicFileName() const
{
  return m_RightIntrinsicEdit->currentPath();
}

//-----------------------------------------------------------------------------
QString StereoCameraCalibrationSelectionWidget::GetLeftToRightTransformationFileName() const
{
  return m_LeftRightTransformEdit->currentPath();
}


//-----------------------------------------------------------------------------
void StereoCameraCalibrationSelectionWidget::SaveDirectoryName(const QString &fullPathName)
{
  QDir pathName(fullPathName);
  m_LastDirectory = pathName.dirName();
}


//-----------------------------------------------------------------------------
void StereoCameraCalibrationSelectionWidget::OnLeftIntrinsicCurrentPathChanged(const QString &path)
{
  SaveDirectoryName(path);
}


//-----------------------------------------------------------------------------
void StereoCameraCalibrationSelectionWidget::OnRightIntrinsicCurrentPathChanged(const QString &path)
{
  SaveDirectoryName(path);
}


//-----------------------------------------------------------------------------
void StereoCameraCalibrationSelectionWidget::OnLeftToRightTransformChanged(const QString &path)
{
  SaveDirectoryName(path);
}

