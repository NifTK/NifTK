/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-18 09:05:48 +0000 (Fri, 18 Nov 2011) $
 Revision          : $Revision: 7804 $
 Last modified by  : $Author: me $

 Original author   : m.espak@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef XnatPluginPreferencePage_h
#define XnatPluginPreferencePage_h

#include "ui_XnatPluginPreferencePage.h"

#include <berryIQtPreferencePage.h>
#include <berryIPreferences.h>

class QWidget;
class QRadioButton;
class QDoubleSpinBox;
class QSpinBox;

/**
 * \class XnatPluginPreferencePage
 * \brief Preferences page for this plugin.
 * \ingroup uk_ac_ucl_cmic_gui_xnat
 *
 */
class XnatPluginPreferencePage : public QObject, public berry::IQtPreferencePage
{
  Q_OBJECT
  Q_INTERFACES(berry::IPreferencePage)

public:
  static const std::string DOWNLOAD_DIRECTORY_NAME;
  static const std::string DOWNLOAD_DIRECTORY_DEFAULT;

  explicit XnatPluginPreferencePage();
  virtual ~XnatPluginPreferencePage();

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

private:

  bool m_Initializing;

  berry::IPreferences::Pointer m_XnatPluginPreferencesNode;

  QWidget* m_MainControl;

  /// \brief All the controls for the main view part.
  Ui::XnatPluginPreferencePage* m_Controls;

  Q_DISABLE_COPY(XnatPluginPreferencePage);
};

#endif
