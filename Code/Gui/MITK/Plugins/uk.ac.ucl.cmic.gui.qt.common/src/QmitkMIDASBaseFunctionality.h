/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-24 15:53:45 +0000 (Thu, 24 Nov 2011) $
 Revision          : $Revision: 7857 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef QMITKMIDASBASEFUNCTIONALITY_H
#define QMITKMIDASBASEFUNCTIONALITY_H

#include <uk_ac_ucl_cmic_gui_qt_common_Export.h>

#include "QmitkAbstractView.h"
#include "mitkSliceNavigationController.h"
#include "mitkILifecycleAwarePart.h"

class QmitkMIDASMultiViewWidget;
class QmitkStdMultiWidget;

/**
 * \class QmitkMIDASBaseFunctionality
 * \brief Base view component for MIDAS plugins providing access to QmitkMIDASMultiViewWidget.
 *
 * \ingroup uk_ac_ucl_cmic_gui_qt_common
 */
class CMIC_QT_COMMON QmitkMIDASBaseFunctionality : public QmitkAbstractView, public mitk::ILifecycleAwarePart
{

  Q_OBJECT

public:

  QmitkMIDASBaseFunctionality();
  QmitkMIDASBaseFunctionality(const QmitkMIDASBaseFunctionality& other);
  virtual ~QmitkMIDASBaseFunctionality();

  /// \brief Derived classes must provide a method to return the view ID.
  virtual std::string GetViewID() const = 0;

  /// \brief Returns the flag indicating whether this view is activated.
  bool IsActivated() const { return m_IsActivated; }

  /// \brief Returns the flag indicating whether this view is visible.
  bool IsVisible() const { return m_IsVisible; }

protected:

  /** \see berry::IPartListener::PartActivated */
  virtual void Activated();

  /** \see berry::IPartListener::PartDeactivated */
  virtual void Deactivated();

  /** \see berry::IPartListener::PartVisible */
  virtual void Visible();

  /** \see berry::IPartListener::PartHidden */
  virtual void Hidden();

  // Callback for when the focus changes, where we update the current render window.
  virtual void OnFocusChanged();

  /// \brief Retrieve the current slice navigation controller from the currently focussed render window, returning NULL if it can't be determined.
  mitk::SliceNavigationController::Pointer GetSliceNavigationController();

  /// \brief Works out the current slice number from the currently focussed render window, returning -1 if it can't be determined.
  int GetSliceNumberFromSliceNavigationController();

  ////////////////////////////////////////////////////////////////////////////////////////////
  /// Should NOT Be Used. Think Very Carefully Before Using This.
  ////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Does a lookup and returns a pointer to the QmitkStdMultiWidget from the editor.
  QmitkStdMultiWidget* GetActiveStdMultiWidget();

  ////////////////////////////////////////////////////////////////////////////////////////////
  /// Should NOT Be Used. Think Very Carefully Before Using This.
  ////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Does a lookup and returns a pointer to the QmitkMIDASMultiViewWidget from the editor.
  QmitkMIDASMultiViewWidget* GetActiveMIDASMultiViewWidget();

  /// \brief Used to store the parent of this view, and should normally be set from withing CreateQtPartControl().
  QWidget *m_Parent;

  /// Stores the activation status.
  bool m_IsActivated;

  /// Stores the visible status.
  bool m_IsVisible;

  // Used for the mitkFocusManager to register callbacks to track the currently focus window.
  unsigned long m_FocusManagerObserverTag;

  // Used to track the focussed renderer.
  mitk::BaseRenderer* m_Focussed2DRenderer;
  mitk::BaseRenderer* m_PreviouslyFocussed2DRenderer;

private:

  // Saves the MITK widget, if available, to avoid repeated lookups.
  QmitkStdMultiWidget *m_MITKWidget;

  // Saves the MIDAS widget, if available, to avoid repeated lookups.
  QmitkMIDASMultiViewWidget *m_MIDASWidget;

};

#endif // QMITKMIDASBASEFUNCTIONALITY_H
