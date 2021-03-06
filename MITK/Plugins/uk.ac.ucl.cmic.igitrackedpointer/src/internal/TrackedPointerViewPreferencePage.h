/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef TrackedPointerViewPreferencePage_h
#define TrackedPointerViewPreferencePage_h

#include <berryIQtPreferencePage.h>
#include <berryIPreferences.h>
#include <QString>

class QWidget;
class QCheckBox;
class ctkPathLineEdit;
class QSpinBox;

/**
 * \class TrackedPointerViewPreferencePage
 * \brief Preferences page for the Tracked Pointer View plugin.
 * \ingroup uk_ac_ucl_cmic_igitrackedpointer_internal
 *
 */
class TrackedPointerViewPreferencePage : public QObject, public berry::IQtPreferencePage
{
  Q_OBJECT
  Q_INTERFACES(berry::IPreferencePage)

public:

  /**
   * \brief Stores the name of the preference node that contains the name of the calibration file.
   */
  static const QString CALIBRATION_FILE_NAME;

  /**
   * \brief Stores the name of the preference node that stores the boolean of whether to update the ortho-view focus point.
   */
  static const QString UPDATE_VIEW_COORDINATE_NAME;

  /**
   * \brief Stores the name of the preference node that stores how many samples to average over.
   */
  static const QString NUMBER_OF_SAMPLES_TO_AVERAGE;

  TrackedPointerViewPreferencePage();
  TrackedPointerViewPreferencePage(const TrackedPointerViewPreferencePage& other);
  ~TrackedPointerViewPreferencePage();

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

private:

  QWidget         *m_MainControl;
  ctkPathLineEdit *m_CalibrationFileName;
  QCheckBox       *m_UpdateViewCoordinate;
  QSpinBox        *m_NumberOfSamplesToAverage;
  bool             m_Initializing;

  berry::IPreferences::Pointer m_TrackedPointerViewPreferencesNode;
};

#endif // TrackedPointerViewPreferencePage_h

