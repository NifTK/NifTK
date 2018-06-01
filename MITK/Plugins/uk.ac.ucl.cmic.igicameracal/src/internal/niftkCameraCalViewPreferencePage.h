/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkCameraCalViewPreferencePage_h
#define niftkCameraCalViewPreferencePage_h

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

  /** Stores the name of the node, not the value of the parameters. */
  static const QString DO_ITERATIVE_NODE_NAME;
  static const QString DO_3D_OPTIMISATION_NODE_NAME;
  static const QString DO_CLUSTERING_NODE_NAME;
  static const QString NUMBER_VIEWS_NODE_NAME;
  static const QString MODEL_NODE_NAME;
  static const QString SCALEX_NODE_NAME;
  static const QString SCALEY_NODE_NAME;
  static const QString PATTERN_NODE_NAME;
  static const QString TAG_FAMILY_NODE_NAME;
  static const QString GRIDX_NODE_NAME;
  static const QString GRIDY_NODE_NAME;
  static const QString HANDEYE_NODE_NAME;
  static const QString REFERENCE_IMAGE_NODE_NAME;
  static const QString REFERENCE_POINTS_NODE_NAME;
  static const QString MODEL_TRANSFORM_NODE_NAME;
  static const QString MINIMUM_NUMBER_POINTS_NODE_NAME;
  static const QString TEMPLATE_IMAGE_NODE_NAME;
  static const QString PREVIOUS_CALIBRATION_DIR_NODE_NAME;
  static const QString OUTPUT_DIR_NODE_NAME;
  static const QString MODEL_IS_STATIONARY_NODE_NAME;
  static const QString CAMERA_IS_STATIONARY_NODE_NAME;
  static const QString SAVE_OUTPUT_BEFORE_CALIBRATION_NODE_NAME;
  static const QString RESET_CALIBRATION_IF_NODE_CHANGES_NODE_NAME;

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

  void OnModelIsStationaryChecked(bool);
  void OnCameraIsStationaryChecked(bool);
  void OnDoIterativeChecked(bool);
  void OnFeaturesComboSelected();

  void On3DModelButtonPressed();
  void OnModelTransformButtonPressed();
  void OnReferenceImageButtonPressed();
  void OnReferencePointsButtonPressed();
  void OnTemplateImageButtonPressed();
  void OnPreviousCalibrationDirButtonPressed();
  void OnOutputDirButtonPressed();

private:

  QScopedPointer<Ui::CameraCalViewPreferencePage> m_Ui;
  QWidget*                                        m_Control;
  bool                                            m_Initializing;
  berry::IPreferences::Pointer                    m_CameraCalViewPreferencesNode;

  void UpdateReferenceImageVisibility();
};

} // end namespace

#endif

