/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef IGIVLEditorPreferencePage_h
#define IGIVLEditorPreferencePage_h

#include <berryIQtPreferencePage.h>
#include <berryIPreferences.h>

class QWidget;
class QRadioButton;
class QPushButton;
class QWidgetAction;
class QCheckBox;
class ctkPathLineEdit;

/**
 * \class IGIVLEditorPreferencePage
 * \brief Preference page for IGIVLEditor.
 * \ingroup uk_ac_ucl_cmic_igivleditor_internal
 */
struct IGIVLEditorPreferencePage : public QObject, public berry::IQtPreferencePage
{
  Q_OBJECT
  Q_INTERFACES(berry::IPreferencePage)

public:
  IGIVLEditorPreferencePage();

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

  berry::IPreferences::Pointer    m_IGIVLEditorPreferencesNode;
};

#endif /* IGIVLEditorPreferencePage_h */
