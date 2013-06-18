/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef SurfaceReconViewPreferencePage_h
#define SurfaceReconViewPreferencePage_h

#include <berryIQtPreferencePage.h>
#include <berryIPreferences.h>
#include <QString>
#include "ui_SurfaceReconViewPreferencePage.h"

class QWidget;
class QPushButton;

/**
 * \class SurfaceReconViewPreferencePage
 * \brief Preferences page for the Surface Reconstruction View plugin.
 * \ingroup uk_ac_ucl_cmic_igisurfacerecon_internal
 *
 */
class SurfaceReconViewPreferencePage : public QObject, public berry::IQtPreferencePage, public Ui::SurfaceReconViewPreferencePageForm
{
  Q_OBJECT
  Q_INTERFACES(berry::IPreferencePage)

public:

  /// \brief Stores the name of the preferences node.
  static const char*      s_PrefsNodeName;
  static const char*      s_DefaultCalibrationFilePathPrefsName;
  static const char*      s_UseUndistortionDefaultPathPrefsName;
  static const char*      s_DefaultTriangulationErrorPrefsName;

  SurfaceReconViewPreferencePage();
  SurfaceReconViewPreferencePage(const SurfaceReconViewPreferencePage& other);
  ~SurfaceReconViewPreferencePage();

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

protected slots:
  void OnDefaultPathBrowseButtonClicked();
  // used for both radio buttons
  void OnUseUndistortRadioButtonClicked();


private:
  berry::IPreferences::Pointer      m_SurfaceReconViewPreferencesNode;
  QString                           m_DefaultCalibrationFilePath;
  bool                              m_UseUndistortPluginDefaultPath;
  float                             m_DefaultTriangulationError;
};

#endif // SurfaceReconViewPreferencePage_h

