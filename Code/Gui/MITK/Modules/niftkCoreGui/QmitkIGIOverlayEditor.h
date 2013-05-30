/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkIGIOverlayEditor_h
#define QmitkIGIOverlayEditor_h

#include "ui_QmitkIGIOverlayEditor.h"
#include "niftkCoreGuiExports.h"
#include <QWidget>
#include <mitkColorProperty.h>
#include <mitkDataStorage.h>

class QmitkRenderWindow;

/**
 * \class QmitkIGIOverlayEditor
 * \brief A widget that contains our QmitkSingle3DView, and a QmitkRenderWindow,
 * (both set to render 3D mode), and several widgets for some basic controls.
 * This class implements all the functionality for IGIOverlayEditor. An
 * additional feature might be to reduce the standalone QmitkRenderWindow in
 * the QmitkIGIOverlayEditor.ui and provide a QmitkStdMultiWidget?
 * \see IGIOverlayEditor
 */
class NIFTKCOREGUI_EXPORT QmitkIGIOverlayEditor : public QWidget, public Ui_QmitkIGIOverlayEditor
{

  Q_OBJECT

public:

  QmitkIGIOverlayEditor(QWidget *parent);
  virtual ~QmitkIGIOverlayEditor();

  //-------------- Start of methods required by IGIOverlayEditor --------------

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
  void SetDepartmentLogoPath(const std::string path);

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
  void SetCalibrationFileName(const std::string& fileName);

  /**
   * \brief Called by framework (event from ctkEventAdmin), to indicate that an update should be performed.
   */
  void Update();

  //-------------- End of methods required by IGIOverlayEditor --------------

private slots:

  void OnOverlayCheckBoxChecked(bool);
  void On3DViewerCheckBoxChecked(bool);
  void OnOpacitySliderMoved(int);
  void OnImageSelected(const mitk::DataNode* node);
  void OnTransformSelected(const mitk::DataNode* node);

private:

  QmitkIGIOverlayEditor(const QmitkIGIOverlayEditor&);  // Purposefully not implemented.
  void operator=(const QmitkIGIOverlayEditor&);  // Purposefully not implemented.
};

#endif // QmitkIGIOverlayEditor_h
