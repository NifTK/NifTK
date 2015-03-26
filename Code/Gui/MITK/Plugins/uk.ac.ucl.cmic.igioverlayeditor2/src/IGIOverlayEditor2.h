/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef IGIOverlayEditor2_h
#define IGIOverlayEditor2_h

#include <QmitkAbstractRenderEditor.h>
#include <mitkILinkedRenderWindowPart.h>
#include <service/event/ctkEvent.h>
#include <uk_ac_ucl_cmic_igioverlayeditor2_Export.h>

class QmitkIGIOverlayEditor2;
class IGIOverlayEditor2Private;

/**
 * \class IGIOverlayEditor
 * \brief Simple editor that delegates all functionality to a QmitkIGIOverlayEditor,
 * and most methods are dummy or do-nothing implementations, as the widget is for
 * a very specific purpose and most of the mitk::ILinkedRenderWindowPart are not needed.
 * \ingroup uk_ac_ucl_cmic_igioverlayeditor
 */
class UK_AC_UCL_CMIC_IGIOVERLAYEDITOR2 IGIOverlayEditor2
    : public QmitkAbstractRenderEditor, public mitk::ILinkedRenderWindowPart
{
  Q_OBJECT

public:

  static const char* EDITOR_ID;

  berryObjectMacro(IGIOverlayEditor2)

  IGIOverlayEditor2();
  ~IGIOverlayEditor2();

  QmitkIGIOverlayEditor2* GetIGIOverlayEditor2();

  // -------------------  mitk::IRenderWindowPart  ----------------------

  /**
   * \see mitk::IRenderWindowPart::GetActiveQmitkRenderWindow()
   */
  QmitkRenderWindow* GetActiveQmitkRenderWindow() const;

  /**
   * \see mitk::IRenderWindowPart::GetQmitkRenderWindows()
   */
  QHash<QString,QmitkRenderWindow*> GetQmitkRenderWindows() const;

  /**
   * \see mitk::IRenderWindowPart::GetQmitkRenderWindow(QString)
   */
  QmitkRenderWindow* GetQmitkRenderWindow(const QString& id) const;

  /**
   * \see mitk::IRenderWindowPart::GetSelectionPosition()
   */
  mitk::Point3D GetSelectedPosition(const QString& id = QString()) const;

  /**
   * \see mitk::IRenderWindowPart::SetSelectedPosition()
   */
  void SetSelectedPosition(const mitk::Point3D& pos, const QString& id = QString());

  /**
   * \see mitk::IRenderWindowPart::EnableDecorations()
   */
  void EnableDecorations(bool enable, const QStringList& decorations = QStringList());

  /**
   * \see mitk::IRenderWindowPart::IsDecorationEnabled()
   */
  bool IsDecorationEnabled(const QString& decoration) const;

  /**
   * \see mitk::IRenderWindowPart::GetDecorations()
   */
  QStringList GetDecorations() const;

  // -------------------  mitk::ILinkedRenderWindowPart  ----------------------

  /**
   * \see mitk::ILinkedRenderWindowPart::GetSlicesRotator()
   */
  mitk::SlicesRotator* GetSlicesRotator() const;

  /**
   * \see mitk::ILinkedRenderWindowPart::GetSlicesSwiveller()
   */
  mitk::SlicesSwiveller* GetSlicesSwiveller() const;

  /**
   * \see mitk::ILinkedRenderWindowPart::EnableSlicingPlanes()
   */
  void EnableSlicingPlanes(bool enable);

  /**
   * \see mitk::ILinkedRenderWindowPart::IsSlicingPlanesEnabled()
   */
  bool IsSlicingPlanesEnabled() const;

  /**
   * \see mitk::ILinkedRenderWindowPart::EnableLinkedNavigation()
   */
  void EnableLinkedNavigation(bool enable);

  /**
   * \see mitk::ILinkedRenderWindowPart::IsLinkedNavigationEnabled()
   */
  bool IsLinkedNavigationEnabled() const;

protected:

  void SetFocus();
  void OnPreferencesChanged(const berry::IBerryPreferences*);
  void CreateQtPartControl(QWidget* parent);

  void WriteCurrentConfig(const QString& directory) const;

protected slots:

  void OnPreferencesChanged();

private slots:

  /**
   * \brief We listen to "uk/ac/ucl/cmic/IGIUPDATE" and call this method.
   */
  void OnIGIUpdate(const ctkEvent& event);

  /**
   * \brief We listen to "uk/ac/ucl/cmic/IGITRACKEDIMAGEUPDATE" and call this method.
   */
  void OnTrackedImageUpdate(const ctkEvent& event);

  /** Listens to "uk/ac/ucl/cmic/IGIRECORDINGSTARTED" on the CTK bus and handles it here. */
  void OnRecordingStarted(const ctkEvent& event);

private:

  const QScopedPointer<IGIOverlayEditor2Private> d;

};

#endif /*IGIOverlayEditor_h */
