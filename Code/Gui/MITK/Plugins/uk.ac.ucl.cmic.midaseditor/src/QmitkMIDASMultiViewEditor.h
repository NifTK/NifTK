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

#include <mitkILinkedRenderWindowPart.h>

#include <QmitkAbstractRenderEditor.h>
#include <mitkDataStorage.h>
#include <mitkRenderingManager.h>
#include <mitkIRenderingManager.h>

#include <uk_ac_ucl_cmic_midaseditor_Export.h>

#include "mitkMIDASEnums.h"
#include "QmitkMIDASMultiViewWidget.h"
#include "QmitkMIDASMultiViewVisibilityManager.h"
#include "mitkMIDASViewKeyPressStateMachine.h"

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
 * \ingroup uk_ac_ucl_cmic_midaseditor
 */

class QmitkMIDASMultiViewEditorPrivate;
class QmitkMIDASMultiViewWidget;
class QmitkRenderWindow;

class MIDASEDITOR_EXPORT QmitkMIDASMultiViewEditor :
  public QmitkAbstractRenderEditor, public mitk::ILinkedRenderWindowPart
{
  Q_OBJECT

public:

  berryObjectMacro(QmitkMIDASMultiViewEditor)

  QmitkMIDASMultiViewEditor();
  ~QmitkMIDASMultiViewEditor();

  static const std::string EDITOR_ID;

  /// \brief Get hold of the internal QmitkMIDASMultiViewWidget.
  QmitkMIDASMultiViewWidget* GetMIDASMultiViewWidget();

  // -------------------  mitk::IRenderWindowPart  ----------------------

  /**
   * \see mitk::IRenderWindowPart::GetActiveRenderWindow()
   */
  virtual QmitkRenderWindow* GetActiveQmitkRenderWindow() const;

  /**
   * \see mitk::IRenderWindowPart::GetRenderWindows()
   */
  virtual QHash<QString,QmitkRenderWindow*> GetQmitkRenderWindows() const;

  /**
   * \see mitk::IRenderWindowPart::GetRenderWindow(QString)
   */
  virtual QmitkRenderWindow* GetQmitkRenderWindow(const QString& id) const;

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

  /**
   * \see mitk::IRenderWindowPart::EnableInteractors().
   */
  void EnableInteractors(bool enable, const QStringList& interactors = QStringList());

  /**
   * \see mitk::IRenderWindowPart::IsInteractorEnabled().
   */
  bool IsInteractorEnabled(const QString& interactor) const;

  /**
   * \see mitk::IRenderWindowPart::GetInteractors().
   */
  QStringList GetInteractors() const;

  // -------------------  mitk::ILinkedRenderWindowPart  ----------------------

  /**
   * \see mitk::ILinkedRenderWindowPart::GetSlicesRotator().
   */
  mitk::SlicesRotator* GetSlicesRotator() const;

  /**
   * \see mitk::ILinkedRenderWindowPart::GetSlicesSwiveller().
   */
  mitk::SlicesSwiveller* GetSlicesSwiveller() const;

  /**
   * \see mitk::ILinkedRenderWindowPart::EnableSlicingPlanes().
   */
  void EnableSlicingPlanes(bool enable);

  /**
   * \see mitk::ILinkedRenderWindowPart::IsSlicingPlanesEnabled().
   */
  bool IsSlicingPlanesEnabled() const;

  /**
   * \see mitk::ILinkedRenderWindowPart::EnableLinkedNavigation().
   */
  void EnableLinkedNavigation(bool enable);

  /**
   * \see mitk::ILinkedRenderWindowPart::IsLinkedNavigationEnabled().
   */
  bool IsLinkedNavigationEnabled() const;

protected:

  /// \brief Tells the contained QmitkMIDASMultiViewWidget to setFocus().
  virtual void SetFocus();

  /// \brief Called when the preferences object of this editor changed.
  virtual void OnPreferencesChanged(const berry::IBerryPreferences*);

  /// \brief Creates the main Qt GUI element parts.
  virtual void CreateQtPartControl(QWidget* parent);

private:

  const QScopedPointer<QmitkMIDASMultiViewEditorPrivate> d;
};

#endif /*QMITKMIDASMULTIVIEWEDITOR_H*/
