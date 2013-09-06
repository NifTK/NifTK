/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef DataSourcesViewPreferencePage_h
#define DataSourcesViewPreferencePage_h

#include <berryIQtPreferencePage.h>
#include <berryIPreferences.h>
#include <QString>
#include <QColor>

class QWidget;
class QPushButton;
class QSpinBox;
class QGridLayout;
class QCheckBox;
class QRadioButton;

class ctkDirectoryButton;

/**
 * \class DataSourcesViewPreferencePage
 * \brief Preferences page for the surgical guidance plugin with choices for colours, default paths to save data etc.
 * \ingroup uk_ac_ucl_cmic_igidatasources_internal
 *
 */
class DataSourcesViewPreferencePage : public QObject, public berry::IQtPreferencePage
{
  Q_OBJECT
  Q_INTERFACES(berry::IPreferencePage)

public:

  /**
   * \brief Stores the name of the preferences node.
   */
  static const std::string PREFERENCES_NODE_NAME;

  DataSourcesViewPreferencePage();
  DataSourcesViewPreferencePage(const DataSourcesViewPreferencePage& other);
  ~DataSourcesViewPreferencePage();

  void Init(berry::IWorkbench::Pointer workbench);

  void CreateQtControl(QWidget* widget);

  QWidget* GetQtControl() const;

  /**
   * \see IPreferencePage::PerformOk()
   */
  virtual bool PerformOk();

  /**
   * \see IPreferencePage::PerformCancel()
   */
  virtual void PerformCancel();

  /**
   * \see IPreferencePage::Update()
   */
  virtual void Update();

private slots:

  void OnErrorColourChanged();
  void OnWarningColourChanged();
  void OnOKColourChanged();

  void OnResetErrorColour();
  void OnResetWarningColour();
  void OnResetOKColour();

private:

  void OnResetColour(int buttonIndex, QColor &color);
  void OnColourChanged(int buttonIndex);
  QGridLayout* CreateColourButtonLayout(QPushButton*& button, QPushButton*& resetButton);

  QWidget        *m_MainControl;

  // We have 3 buttons... 0=Error, 1=Warning, 2=OK
  QPushButton    *m_ColorPushButton[3];
  QPushButton    *m_ColorResetPushButton[3];
  QString         m_ColorStyleSheet[3];
  std::string     m_Color[3];

  // Other controls.
  QSpinBox           *m_FramesPerSecondSpinBox;
  QSpinBox           *m_ClearDataSpinBox;
  QSpinBox           *m_MilliSecondsTolerance;
  ctkDirectoryButton *m_DirectoryPrefix;
  QRadioButton       *m_SaveOnUpdate;
  QRadioButton       *m_SaveOnReceive;
  QCheckBox          *m_SaveInBackground;
  QCheckBox          *m_PickLatestData;

  bool m_Initializing;

  berry::IPreferences::Pointer m_DataSourcesViewPreferencesNode;
};

#endif // DataSourcesViewPreferencePage_h

