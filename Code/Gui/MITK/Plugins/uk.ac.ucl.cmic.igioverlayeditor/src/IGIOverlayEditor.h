/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef IGIOverlayEditor_h
#define IGIOverlayEditor_h

#include <QmitkAbstractRenderEditor.h>
#include <mitkILinkedRenderWindowPart.h>

#include <uk_ac_ucl_cmic_igioverlayeditor_Export.h>

class QmitkIGIOverlayEditor;
class IGIOverlayEditorPrivate;

/**
 * \class IGIOverlayEditor
 * \brief Simple editor that delegates all functionality to a QmitkIGIOverlayEditor.
 * \ingroup uk_ac_ucl_cmic_igioverlayeditor
 */
class UK_AC_UCL_CMIC_IGIOVERLAYEDITOR IGIOverlayEditor
    : public QmitkAbstractRenderEditor, public mitk::ILinkedRenderWindowPart
{
  Q_OBJECT

public:

  static const std::string EDITOR_ID;

  berryObjectMacro(IGIOverlayEditor)

  IGIOverlayEditor();
  ~IGIOverlayEditor();

  QmitkIGIOverlayEditor* GetIGIOverlayEditor();

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

private:

  const QScopedPointer<IGIOverlayEditorPrivate> d;

};

#endif /*IGIOverlayEditor_h */
