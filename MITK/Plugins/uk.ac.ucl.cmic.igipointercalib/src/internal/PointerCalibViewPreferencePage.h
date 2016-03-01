/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef PointerCalibViewPreferencePage_h
#define PointerCalibViewPreferencePage_h

#include <berryIQtPreferencePage.h>
#include <berryIPreferences.h>
#include <QString>

class QWidget;
class QCheckBox;
class ctkPathLineEdit;
class QSpinBox;

/**
 * \class PointerCalibViewPreferencePage
 * \brief Preferences page for the Ultrasound Pointer Calibration View plugin.
 * \ingroup uk_ac_ucl_cmic_igipointercalib_internal
 *
 */
class PointerCalibViewPreferencePage : public QObject, public berry::IQtPreferencePage
{
  Q_OBJECT
  Q_INTERFACES(berry::IPreferencePage)

public:

  /**
   * \brief Stores the name of the preferences node.
   */
  static const std::string PREFERENCES_NODE_NAME;

  PointerCalibViewPreferencePage();
  PointerCalibViewPreferencePage(const PointerCalibViewPreferencePage& other);
  ~PointerCalibViewPreferencePage();

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

private:

  QWidget         *m_MainControl;
  bool             m_Initializing;

  berry::IPreferences::Pointer m_PointerCalibViewPreferencesNode;
};

#endif // PointerCalibViewPreferencePage_h

