/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QMITKCOMMONAPPSAPPLICATIONPREFERENCEPAGE_H_
#define QMITKCOMMONAPPSAPPLICATIONPREFERENCEPAGE_H_

#include <uk_ac_ucl_cmic_gui_qt_commonapps_Export.h>
#include <berryIQtPreferencePage.h>
#include <berryIPreferences.h>

class QWidget;
class QComboBox;
class QCheckBox;
class QDoubleSpinBox;

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
  static const std::string BINARY_OPACITY_NAME;
  static const double BINARY_OPACITY_VALUE;

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
  QDoubleSpinBox *m_BinaryOpacity;

  berry::IPreferences::Pointer m_PreferencesNode;
};

#endif /* QMITKCOMMONAPPSAPPLICATIONPREFERENCEPAGE_H_ */

