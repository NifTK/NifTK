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

#if 0
  static const std::string AUTO_UPDATE_NAME;
  static const std::string ASSUME_BINARY_NAME;
  static const std::string REQUIRE_SAME_SIZE_IMAGE_NAME;
  static const std::string BACKGROUND_VALUE_NAME;
#endif

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

protected:
#if 0

  QWidget*   m_MainControl;
  QCheckBox* m_AutoUpdate;
  QCheckBox* m_AssumeBinary;
  QCheckBox* m_RequireSameSizeImage;
  QSpinBox*  m_BackgroundValue;
  berry::IPreferences::Pointer m_ImageStatisticsPreferencesNode;
#endif
};

#endif /* UndistortViewPreferencesPage */

