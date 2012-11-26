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

#ifndef QMITKCOMMONAPPSAPPLICATIONPREFERENCEPAGE_H_
#define QMITKCOMMONAPPSAPPLICATIONPREFERENCEPAGE_H_

#include "mitkQtCommonAppsAppDll.h"

#include "berryIQtPreferencePage.h"
#include <berryIPreferences.h>

class QWidget;
class QComboBox;
class QCheckBox;

/**
 * \class QmitkCommonAppsApplicationPreferencePage
 * \brief Preferences page for this plugin, providing application wide defaults.
 * \ingroup uk_ac_ucl_cmic_gui_qt_commonapps_internal
 *
 */
class CMIC_QT_COMMONAPPS QmitkCommonAppsApplicationPreferencePage : public QObject, public berry::IQtPreferencePage
{
  Q_OBJECT
  Q_INTERFACES(berry::IPreferencePage)

public:

  QmitkCommonAppsApplicationPreferencePage();
  QmitkCommonAppsApplicationPreferencePage(const QmitkCommonAppsApplicationPreferencePage& other);
  ~QmitkCommonAppsApplicationPreferencePage();

  static const std::string IMAGE_RESLICE_INTERPOLATION;
  static const std::string IMAGE_TEXTURE_INTERPOLATION;
  static const std::string BLACK_OPACITY;

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

  QWidget        *m_MainControl;
  QComboBox      *m_ResliceInterpolation;
  QComboBox      *m_TextureInterpolation;
  QCheckBox      *m_BlackOpacity;

  berry::IPreferences::Pointer m_PreferencesNode;
};

#endif /* QMITKCOMMONAPPSAPPLICATIONPREFERENCEPAGE_H_ */

