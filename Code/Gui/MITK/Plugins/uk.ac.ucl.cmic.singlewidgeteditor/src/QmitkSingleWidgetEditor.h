/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QMITKSINGLEWIDGETEDITOR_H_
#define QMITKSINGLEWIDGETEDITOR_H_

#include <QmitkAbstractRenderEditor.h>

#include <mitkILinkedRenderWindowPart.h>

#include <uk_ac_ucl_cmic_singlewidgeteditor_Export.h>

class QmitkSingleWidget;
class QmitkMouseModeSwitcher;
class QmitkSingleWidgetEditorPrivate;

/**
 * \ingroup uk_ac_ucl_cmic_singlewidgeteditor
 */
class UK_AC_UCL_CMIC_SINGLEWIDGETEDITOR QmitkSingleWidgetEditor
    : public QmitkAbstractRenderEditor, public mitk::ILinkedRenderWindowPart
{
  Q_OBJECT

public:

  berryObjectMacro(QmitkSingleWidgetEditor)

  static const std::string EDITOR_ID;

  QmitkSingleWidgetEditor();
  ~QmitkSingleWidgetEditor();

  QmitkSingleWidget* GetSingleWidget();

  /**
   * Request the QmitkRenderWindowMenus to be either off, or whatever was the last known state, which is
   * useful when responding to the PartOpened, PartClosed, PartHidden methods.
   *
   * \param on If <code>true</code> will request the QmitkSingelWidget to set the QmitkRenderWindowMenu to
   *           whatever was the last known state, and if <code>false</code> will turn the QmitkRenderWindowMenu off.
   *
   */
  void RequestActivateMenuWidget(bool on);

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

  mitk::SlicesRotator* GetSlicesRotator() const;
  mitk::SlicesSwiveller* GetSlicesSwiveller() const;

  void EnableSlicingPlanes(bool enable);
  bool IsSlicingPlanesEnabled() const;

  void EnableLinkedNavigation(bool enable);
  bool IsLinkedNavigationEnabled() const;

protected:

  void SetFocus();

  void OnPreferencesChanged(const berry::IBerryPreferences*);

  void CreateQtPartControl(QWidget* parent);

private:

  const QScopedPointer<QmitkSingleWidgetEditorPrivate> d;

};

#endif /*QMITKSINGLEWIDGETEDITOR_H_*/
