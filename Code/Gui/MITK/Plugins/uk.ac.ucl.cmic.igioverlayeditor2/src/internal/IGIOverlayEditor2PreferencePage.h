/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef IGIOverlayEditor2PreferencePage_h
#define IGIOverlayEditor2PreferencePage_h

#include <berryIQtPreferencePage.h>
#include <berryIPreferences.h>

class QWidget;
class QRadioButton;
class QPushButton;
class QWidgetAction;
class QCheckBox;
class ctkPathLineEdit;

/**
 * \class IGIOverlayEditor2PreferencePage
 * \brief Preference page for IGIOverlayEditor2.
 * \ingroup uk_ac_ucl_cmic_igioverlayeditor2_internal
 */
struct IGIOverlayEditor2PreferencePage : public QObject, public berry::IQtPreferencePage
{
  Q_OBJECT
  Q_INTERFACES(berry::IPreferencePage)

public:
  IGIOverlayEditor2PreferencePage();

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


  static const char*          BACKGROUND_COLOR_PREFSKEY;
  static const unsigned int   DEFAULT_BACKGROUND_COLOR;

public slots:
  void OnBackgroundColourClicked();

protected:

  QWidget*        m_MainControl;
  QPushButton*    m_BackgroundColourButton;
  unsigned int    m_BackgroundColour;

  berry::IPreferences::Pointer    m_IGIOverlayEditor2PreferencesNode;
};

#endif /* IGIOverlayEditor2PreferencePage_h */
