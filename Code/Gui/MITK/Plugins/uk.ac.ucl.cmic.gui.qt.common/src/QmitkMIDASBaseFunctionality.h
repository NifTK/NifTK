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

#include "QmitkFunctionality.h"
#include "berryIPartListener.h"
#include <berryISelectionListener.h>
#include <QWidget>
#include "mitkDataStorage.h"

class QmitkMIDASMultiViewWidget;

/**
 * \class QmitkMIDASBaseFunctionality
 * \brief Base view component for MIDAS plugins handling creating, connecting to
 * and disconnecting from the QmitkMIDASMultiViewWidget, to give the MIDAS style layouts.
 *
 * \ingroup uk_ac_ucl_cmic_gui_qt_common
 */
class CMIC_QT_COMMON QmitkMIDASBaseFunctionality : public QmitkFunctionality
{

  Q_OBJECT

public:

  QmitkMIDASBaseFunctionality();
  QmitkMIDASBaseFunctionality(const QmitkMIDASBaseFunctionality& other);
  virtual ~QmitkMIDASBaseFunctionality();

  /// \brief Derived classes must provide a method to return the view ID.
  virtual std::string GetViewID() const = 0;

  /// \brief Called by framework to create the plugins controls.
  virtual void CreateQtPartControl(QWidget *parent);

  /// \brief Required implementation, inherited from base classes, currently does nothing.
  virtual void SetFocus();

  /// \brief When this plugin is activated we setup the datastore.
  virtual void Activated();

  /// \brief Called when this plugin is deactivated, currently does nothing.
  virtual void Deactivated();

  /// \brief Inject the widget pointer into this class, this gets called by m_MIDASMultiViewWidgetListener when the widget part is opened/closed.
  virtual void SetMIDASMultiViewWidget(QmitkMIDASMultiViewWidget *widget);

  /// \brief Returns the default data storage.
  virtual mitk::DataStorage::Pointer GetDefaultDataStorage() const;

protected:

  /// \brief Saves the parent of this view.
  QWidget* m_Parent;

  /// \brief Does a lookup and returns a pointer to the QmitkMIDASMultiViewWidget from the editor.
  QmitkMIDASMultiViewWidget* GetActiveMIDASMultiViewWidget();

private:

  // Each derived class will have access to this pointer, which will be populated when the part opens.
  QmitkMIDASMultiViewWidget *m_MIDASMultiViewWidget;

  // This listener is reponsible for looking up the editor, and getting hold of the QmitkMIDASMultiViewWidget.
  berry::IPartListener::Pointer m_MIDASMultiViewWidgetListener;
};

#endif // QMITKMIDASBASEFUNCTIONALITY_H
