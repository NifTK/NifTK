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
 * \brief Implements an Editor containing an overlay widget, and a 3D rendering widget, and controls.
 */
class NIFTKCOREGUI_EXPORT QmitkIGIOverlayEditor : public QWidget, public Ui_QmitkIGIOverlayEditor
{

  Q_OBJECT

public:

  QmitkIGIOverlayEditor(QWidget *parent);
  virtual ~QmitkIGIOverlayEditor();

  //-------------- Start of methods required by IGIOverlayEditor --------------

  void SetDataStorage(mitk::DataStorage* storage);

  QmitkRenderWindow* GetActiveQmitkRenderWindow() const;

  QHash<QString, QmitkRenderWindow *> GetQmitkRenderWindows() const;

  QmitkRenderWindow* GetQmitkRenderWindow(const QString &id) const;

  mitk::Point3D GetSelectedPosition(const QString &id) const;

  void SetSelectedPosition(const mitk::Point3D &pos, const QString &id);

  void SetDepartmentLogoPath(const std::string path);

  void EnableDepartmentLogo();

  void DisableDepartmentLogo();

  void SetGradientBackgroundColors(const mitk::Color& colour1, const mitk::Color& colour2);

  void EnableGradientBackground();

  void DisableGradientBackground();

  //-------------- End of methods required by IGIOverlayEditor --------------

public slots:

  void OnOverlayCheckBoxChecked(bool);
  void On3DViewerCheckBoxChecked(bool);

private:

  QmitkIGIOverlayEditor(const QmitkIGIOverlayEditor&);  // Purposefully not implemented.
  void operator=(const QmitkIGIOverlayEditor&);  // Purposefully not implemented.
};

#endif // QmitkIGIOverlayEditor_h
