/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkIGIOverlayEditorWidget_h
#define niftkIGIOverlayEditorWidget_h

#include "ui_niftkIGIOverlayEditorWidget.h"
#include "niftkIGIOverlayEditorExports.h"
#include <QWidget>
#include <mitkColorProperty.h>
#include <mitkDataStorage.h>

class QmitkRenderWindow;

namespace niftk
{
/**
 * \class IGIOverlayEditorWidget
 * \brief A widget that contains our QmitkSingle3DView, and a QmitkRenderWindow,
 * (both set to render 3D mode), and several widgets for some basic controls.
 * This class implements all the functionality for IGIOverlayEditorWidget. An
 * additional feature might be to remove the standalone QmitkRenderWindow in
 * the niftkIGIOverlayEditorWidget.ui and provide a QmitkStdMultiWidget?
 * \see IGIOverlayEditorWidget
 */
class NIFTKIGIOVERLAYEDITOR_EXPORT IGIOverlayEditorWidget : public QWidget,
    public Ui_IGIOverlayEditorWidget
{

  Q_OBJECT

public:

  IGIOverlayEditorWidget(QWidget *parent);
  virtual ~IGIOverlayEditorWidget();

  //-------------- Start of methods required by IGIOverlayEditorWidget --------------

  void SetDataStorage(mitk::DataStorage* storage);

  /**
   * \brief Currently returns the QmitkRenderWindow from the QmitkSingle3DView.
   *
   * In future we could pick whichever one has focus or something.
   */
  QmitkRenderWindow* GetActiveQmitkRenderWindow() const;

  /**
   * \brief Returns the QmitkSingle3DView's QmitkRenderWindow with identifier="overlay",
   * and the 3D QmitkRenderWindow with identifier="3d".
   */
  QHash<QString, QmitkRenderWindow *> GetQmitkRenderWindows() const;

  /**
   * \brief Returns the QmitkRenderWindow corresponding to the parameter id.
   * \param id identifer, which for this class must be either "overlay" or "3d".
   * \return QmitkRenderWindow* or NULL if the id does not match.
   */
  QmitkRenderWindow* GetQmitkRenderWindow(const QString &id) const;

  /**
   * \brief Set the full path for the department logo, currently delegating only to QmitkSingle3DView.
   */
  void SetDepartmentLogoPath(const QString& path);

  /**
   * \brief Calls QmitkSingle3DView::EnableDepartmentLogo().
   */
  void EnableDepartmentLogo();

  /**
   * \brief Calls QmitkSingle3DView::DisableDepartmentLogo().
   */
  void DisableDepartmentLogo();

  /**
   * \brief Calls QmitkSingle3DView::SetGradientBackgroundColors().
   */
  void SetGradientBackgroundColors(const mitk::Color& colour1, const mitk::Color& colour2);

  /**
   * \brief Calls QmitkSingle3DView::EnableGradientBackground().
   */
  void EnableGradientBackground();

  /**
   * \brief Calls QmitkSingle3DView::DisableGradientBackground().
   */
  void DisableGradientBackground();

  /**
   * \brief Sets the calibration file (e.g. hand-eye transform for a laparoscope).
   */
  void SetCalibrationFileName(const QString& fileName);

  QString GetCalibrationFileName() const;

  /**
   * \brief Sets whether or not we are doing camera tracking mode.
   */
  void SetCameraTrackingMode(const bool& isCameraTracking);

  /**
   * \brief Sets whether or not we clip to the image plane when we are in image tracking mode.
   */
  void SetClipToImagePlane(const bool& clipToImagePlane);

  /**
   * \brief Called by framework (event from ctkEventAdmin), to indicate that an update should be performed.
   */
  void Update();

  //-------------- End of methods required by IGIOverlayEditorWidget --------------

private slots:

  void OnOverlayCheckBoxChecked(bool);
  void On3DViewerCheckBoxChecked(bool);
  void OnOpacitySliderMoved(int);
  void OnImageSelected(const mitk::DataNode* node);
  void OnTransformSelected(const mitk::DataNode* node);

private:

  IGIOverlayEditorWidget(const IGIOverlayEditorWidget&);  // Purposefully not implemented.
  void operator=(const IGIOverlayEditorWidget&);  // Purposefully not implemented.

  /**
   * \brief Utility method to deregister data storage listeners.
   */
  void DeRegisterDataStorageListeners();

  /**
   * \brief Called when a DataStorage Node Changed Event was emitted.
   */
  void NodeChanged(const mitk::DataNode* node);

  mitk::DataStorage::Pointer m_DataStorage;
};

} // end namespace

#endif
