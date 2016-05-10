/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef CameraCalViewPreferencePage_h
#define CameraCalViewPreferencePage_h

#include <berryIQtPreferencePage.h>
#include <berryIPreferences.h>
#include <QString>
#include <QScopedPointer>

class QWidget;

namespace Ui
{
class CameraCalViewPreferencePage;
}

namespace niftk
{

/**
 * \class CameraCalViewPreferencePage
 * \brief Preferences page for the Video Camera Calibration View plugin.
 * \ingroup uk_ac_ucl_cmic_igicameracal_internal
 *
 */
class CameraCalViewPreferencePage : public QObject, public berry::IQtPreferencePage
{
  Q_OBJECT
  Q_INTERFACES(berry::IPreferencePage)

public:

  /**
   * \brief Stores the name of the preferences node.
   */
  static const QString PREFERENCES_NODE_NAME;

  CameraCalViewPreferencePage();
  CameraCalViewPreferencePage(const CameraCalViewPreferencePage& other);
  ~CameraCalViewPreferencePage();

  void Init(berry::IWorkbench::Pointer workbench) override;

  void CreateQtControl(QWidget* widget) override;

  QWidget* GetQtControl() const override;

  /**
   * \see IPreferencePage::PerformOk()
   */
  virtual bool PerformOk() override;

  /**
   * \see IPreferencePage::PerformCancel()
   */
  virtual void PerformCancel() override;

  /**
   * \see IPreferencePage::Update()
   */
  virtual void Update() override;

private slots:

  void OnFeaturesComboSelected();
  void On3DModelButtonPressed();
  void OnModelToTrackerButtonPressed();
  void OnOutputDirectoryButtonPressed();

private:

  QScopedPointer<Ui::CameraCalViewPreferencePage> m_Ui;
  QWidget*                                        m_Control;
  bool                                            m_Initializing;
  berry::IPreferences::Pointer                    m_CameraCalViewPreferencesNode;
};

} // end namespace

#endif

