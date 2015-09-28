/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef StereoCameraCalibrationSelectionWidget_h
#define StereoCameraCalibrationSelectionWidget_h

#include "niftkIGIGuiExports.h"
#include "ui_StereoCameraCalibrationSelectionWidget.h"
#include <QWidget>
#include <QString>

/**
 * \class StereoCameraCalibrationSelectionWidget
 * \brief Used to group widgets to specify pathnames for left/right intrinsics
 * and left-to-right transformation for a stereo pair.
 */
class NIFTKIGIGUI_EXPORT StereoCameraCalibrationSelectionWidget : public QWidget, public Ui_StereoCameraCalibrationSelectionWidget
{
  Q_OBJECT

public:
  StereoCameraCalibrationSelectionWidget(QWidget *parent = 0);

  QString GetLastDirectory() const;
  void SetLastDirectory(const QString& directoryName);

  QString GetLeftIntrinsicFileName() const;
  QString GetRightIntrinsicFileName() const;
  QString GetLeftToRightTransformationFileName() const;

  void SetRightChannelEnabled(const bool& isEnabled);
  void SetLeftChannelEnabled(const bool& isEnabled);

private slots:

  void OnLeftIntrinsicCurrentPathChanged(const QString &path);
  void OnRightIntrinsicCurrentPathChanged(const QString &path);
  void OnLeftToRightTransformChanged(const QString &path);

private:

  void SaveDirectoryName(const QString &fullPathName);

  /**
   * \brief Used to store the last directory accessed.
   */
  QString m_LastDirectory;

};

#endif // StereoCameraCalibrationSelectionWidget_h
