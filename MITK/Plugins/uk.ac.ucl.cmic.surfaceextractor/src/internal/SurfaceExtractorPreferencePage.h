/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef SurfaceExtractorPreferencePage_h
#define SurfaceExtractorPreferencePage_h

#include <QObject>
#include <berryIQtPreferencePage.h>

class QWidget;

class SurfaceExtractorPreferencePagePrivate;

/**
 * \class SurfaceExtractorPreferencePage
 * \brief Preferences page for this plugin, to set defaults for the Surface Extractor view.
 * \ingroup uk_ac_ucl_cmic_gui_surfaceextractor
 *
 */
class SurfaceExtractorPreferencePage : public QObject, public berry::IQtPreferencePage
{
  Q_OBJECT
  Q_INTERFACES(berry::IPreferencePage)

public:

  SurfaceExtractorPreferencePage();
  ~SurfaceExtractorPreferencePage();

  static const QString GAUSSIAN_SMOOTH_NAME;
  static const bool GAUSSIAN_SMOOTH_DEFAULT;
  static const QString GAUSSIAN_STDDEV_NAME;
  static const double GAUSSIAN_STDDEV_DEFAULT;
  static const QString THRESHOLD_NAME;
  static const double THRESHOLD_DEFAULT;
  static const QString TARGET_REDUCTION_NAME;
  static const double TARGET_REDUCTION_DEFAULT;
  static const QString MAX_NUMBER_OF_POLYGONS_NAME;
  static const long MAX_NUMBER_OF_POLYGONS_DEFAULT;

  virtual void Init(berry::IWorkbench::Pointer workbench) override;

  void CreateQtControl(QWidget* widget) override;

  QWidget* GetQtControl() const override;

  ///
  /// \see IPreferencePage::PerformOk()
  ///
  virtual bool PerformOk() override;

  ///
  /// \see IPreferencePage::PerformCancel()
  ///
  virtual void PerformCancel() override;

  ///
  /// \see IPreferencePage::Update()
  ///
  virtual void Update() override;

private slots:
  void on_cbxGaussianSmooth_toggled(bool);

private:
  QWidget* ui;
  QScopedPointer<SurfaceExtractorPreferencePagePrivate> d_ptr;

  Q_DECLARE_PRIVATE(SurfaceExtractorPreferencePage);
  Q_DISABLE_COPY(SurfaceExtractorPreferencePage);
};

#endif /* _SurfaceExtractorPreferencePage_h */
