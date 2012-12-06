/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-18 09:05:48 +0000 (Fri, 18 Nov 2011) $
 Revision          : $Revision: 7804 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef _SURGICALGUIDANCEVIEWPREFERENCEPAGE_H_INCLUDED
#define _SURGICALGUIDANCEVIEWPREFERENCEPAGE_H_INCLUDED

#include "berryIQtPreferencePage.h"
#include <berryIPreferences.h>
#include <QString>
#include <QColor>

class QWidget;
class QPushButton;
class QSpinBox;
class QGridLayout;
class ctkDirectoryButton;

/**
 * \class SurgicalGuidanceViewPreferencePage
 * \brief Preferences page for the surgical guidance plugin with choices for colours, default paths to save data etc.
 * \ingroup uk_ac_ucl_cmic_surgicalguidance_internal
 *
 */
class SurgicalGuidanceViewPreferencePage : public QObject, public berry::IQtPreferencePage
{
  Q_OBJECT
  Q_INTERFACES(berry::IPreferencePage)

public:

  /// \brief Stores the name of the preferences node.
  static const std::string PREFERENCES_NODE_NAME;

  SurgicalGuidanceViewPreferencePage();
  SurgicalGuidanceViewPreferencePage(const SurgicalGuidanceViewPreferencePage& other);
  ~SurgicalGuidanceViewPreferencePage();

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
  ctkDirectoryButton *m_DirectoryPrefix;

  bool m_Initializing;

  berry::IPreferences::Pointer m_SurgicalGuidanceViewPreferencesNode;
};

#endif /* _SURGICALGUIDANCEVIEWPREFERENCEPAGE_H_INCLUDED */

