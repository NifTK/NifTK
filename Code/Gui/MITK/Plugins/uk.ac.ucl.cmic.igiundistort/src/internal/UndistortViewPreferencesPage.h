/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef UndistortViewPreferencesPage_h
#define UndistortViewPreferencesPage_h

#include <berryIQtPreferencePage.h>
#include <berryIPreferences.h>
#include "ui_UndistortViewPreferencePage.h"

class QWidget;
class QCheckBox;
class QSpinBox;


class UndistortViewPreferencesPage : public QObject, public berry::IQtPreferencePage, public Ui::UndistortViewPreferencePageForm
{
  Q_OBJECT
  Q_INTERFACES(berry::IPreferencePage)

public:

  UndistortViewPreferencesPage();
  UndistortViewPreferencesPage(const UndistortViewPreferencesPage& other);
  ~UndistortViewPreferencesPage();

  static const char*      s_DefaultCalibrationFilePathPrefsName;


  void Init(berry::IWorkbench::Pointer workbench);

  void CreateQtControl(QWidget* widget);

  QWidget* GetQtControl() const;

  ///
  /// \see IPreferencePage::PerformOk()
  ///
  virtual bool PerformOk();

  ///
  /// \see IPreferencePage::PerformCancel()
  ///
  virtual void PerformCancel();

  ///
  /// \see IPreferencePage::Update()
  ///
  virtual void Update();

protected slots:
  void OnDefaultPathBrowseButtonClicked();

protected:
  berry::IPreferences::Pointer      m_UndistortPreferencesNode;

  QString                           m_DefaultCalibrationFilePath;
};

#endif /* UndistortViewPreferencesPage */

