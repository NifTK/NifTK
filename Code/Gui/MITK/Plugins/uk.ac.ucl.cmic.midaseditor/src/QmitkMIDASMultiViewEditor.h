/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-12-05 16:22:23 +0000 (Mon, 05 Dec 2011) $
 Revision          : $Revision: 7920 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef QMITKMIDASMULTIVIEWEDITOR_H
#define QMITKMIDASMULTIVIEWEDITOR_H

#include <berryQtEditorPart.h>
#include <berryIPartListener.h>
#include <berryIPreferences.h>

#include <berryIPreferencesService.h>
#include <berryIBerryPreferences.h>
#include <berryISelection.h>
#include <berryISelectionProvider.h>
#include <berryISelectionListener.h>

#include <uk_ac_ucl_cmic_midaseditor_Export.h>

#include "QmitkAbstractRenderEditor.h"
#include "QmitkMIDASViewEnums.h"
#include "QmitkMIDASMultiViewWidget.h"
#include "QmitkMIDASMultiViewVisibilityManager.h"
#include "mitkDataStorage.h"
#include "mitkRenderingManager.h"
#include "mitkIRenderingManager.h"
#include "mitkMIDASKeyPressStateMachine.h"

namespace mitk {
  class DataNode;
}

/**
 * \class QmitkMIDASMultiViewEditor
 * \brief Provides a MIDAS style layout, with up to 5 x 5 panes of equal size in a grid layout.
 *
 * As of 18th April 2012, this editor inherits from the QmitkAbstractRenderEditor, and hence
 * conforms to the mitk::IRenderWindowPart which is the new Render Window Abstraction provided by
 * MITK on 24.02.2012, apart from the decorations. This editor purposefully implements the methods
 * EnableDecorations, IsDecorationEnabled, GetDecorations to do nothing (see method documentation).
 *
 * \ingroup uk_ac_ucl_cmic_gui_qt_common
 */
class MIDASEDITOR_EXPORT QmitkMIDASMultiViewEditor :
  public QmitkAbstractRenderEditor, virtual public berry::IPartListener
{
  Q_OBJECT

public:

  berryObjectMacro(QmitkMIDASMultiViewEditor)

  QmitkMIDASMultiViewEditor();
  QmitkMIDASMultiViewEditor(const QmitkMIDASMultiViewEditor& other);
  ~QmitkMIDASMultiViewEditor();

  static const std::string EDITOR_ID;

  /// \brief Get hold of the internal QmitkMIDASMultiViewWidget.
  QmitkMIDASMultiViewWidget* GetMIDASMultiViewWidget();

  // -------------------  mitk::IRenderWindowPart  ----------------------

  /**
   * \see mitk::IRenderWindowPart::GetActiveRenderWindow()
   */
  virtual QmitkRenderWindow* GetActiveRenderWindow() const;

  /**
   * \see mitk::IRenderWindowPart::GetRenderWindows()
   */
  virtual QHash<QString,QmitkRenderWindow*> GetRenderWindows() const;

  /**
   * \see mitk::IRenderWindowPart::GetRenderWindow(QString)
   */
  virtual QmitkRenderWindow* GetRenderWindow(const QString& id) const;

  /**
   * \see mitk::IRenderWindowPart::GetSelectionPosition()
   */
  virtual mitk::Point3D GetSelectedPosition(const QString& id = QString()) const;

  /**
   * \see mitk::IRenderWindowPart::SetSelectedPosition()
   */
  virtual void SetSelectedPosition(const mitk::Point3D& pos, const QString& id = QString());

  /**
   * \see mitk::IRenderWindowPart::EnableDecorations(), and in this class, deliberately implemented as a no-op.
   */
  virtual void EnableDecorations(bool enable, const QStringList& decorations = QStringList());

  /**
   * \see mitk::IRenderWindowPart::IsDecorationEnabled(), and in this class, only returns false.
   */
  virtual bool IsDecorationEnabled(const QString& decoration) const;

  /**
   * \see mitk::IRenderWindowPart::GetDecorations(), and in this class, always returns empty list.
   */
  virtual QStringList GetDecorations() const;

  /**
   * Get the RenderingManager used by this editor. This default implementation uses the
   * global MITK RenderingManager provided by mitk::RenderingManager::GetInstance().
   *
   * \see mitk::IRenderWindowPart::GetRenderingManager
   */
  virtual mitk::IRenderingManager* GetRenderingManager() const;

protected:

  /// \brief Tells the contained QmitkMIDASMultiViewWidget to setFocus().
  virtual void SetFocus();

  // Creates the main Qt GUI element parts.
  virtual void CreateQtPartControl(QWidget* parent);

  /// \brief Called when the preferences object of this editor changed.
  virtual void OnPreferencesChanged(const berry::IBerryPreferences*);

  // -------------------  mitk::IPartListener  ----------------------

  Events::Types GetPartEventTypes() const;
  virtual void PartClosed (berry::IWorkbenchPartReference::Pointer partRef);
  virtual void PartHidden (berry::IWorkbenchPartReference::Pointer partRef);
  virtual void PartVisible (berry::IWorkbenchPartReference::Pointer partRef);

private:

  // This class hooks into the Global Interaction system to respond to Key press events.
  mitk::MIDASKeyPressStateMachine::Pointer m_KeyPressStateMachine;

  // This class is the main central widget, containing multiple widgets such as rendering windows and control buttons.
  QmitkMIDASMultiViewWidget* m_MIDASMultiViewWidget;

  // This class is to manage visibility when nodes added, removed, main visibility properties changed etc. and manage the renderer specific properties.
  QmitkMIDASMultiViewVisibilityManager* m_MidasMultiViewVisibilityManager;

  // We maintain our own RenderingManager. NOTE: It's NOT the Global one.
  mitk::RenderingManager::Pointer m_RenderingManager;

};

#endif /*QMITKMIDASMULTIVIEWEDITOR_H*/
