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

#include "berryIPartListener.h"
#include "berryQtViewPart.h"
#include <QWidget>
#include "mitkDataStorage.h"

class QmitkMIDASMultiViewWidget;

/**
 * \class QmitkMIDASBaseFunctionality
 * \brief Base view component for MIDAS plugins handling creating, connecting to
 * and disconnecting from the QmitkMIDASMultiViewWidget, to give the MIDAS style layouts.
 * \ingroup uk_ac_ucl_cmic_gui_qt_common
 * \sa QmitkCMICBaseFunctionality
 */
class CMIC_QT_COMMON QmitkMIDASBaseFunctionality : public berry::QtViewPart
{

  Q_OBJECT

public:

  QmitkMIDASBaseFunctionality();
  QmitkMIDASBaseFunctionality(const QmitkMIDASBaseFunctionality& other);
  virtual ~QmitkMIDASBaseFunctionality();

  virtual void CreateQtPartControl(QWidget *parent);
  virtual void SetMIDASMultiViewWidget(QmitkMIDASMultiViewWidget *widget);
  virtual std::string GetViewID() const = 0;
  virtual void SetFocus();

  /// \brief When this plugin is activated we setup the datastore.
  virtual void Activated();

protected slots:

protected:

  /// \brief Saves the parent of this view (this is the scrollarea created in CreatePartControl(void*)
  QWidget* m_Parent;

  /// \Returns a pointer to the MIDASMultiViewWidget from the editor.
  QmitkMIDASMultiViewWidget* GetActiveMIDASMultiViewWidget();

  /// \brief We store a reference to the single data storage, and populate it when this plugin is Activated.
  mitk::DataStorage::Pointer m_DataStorage;

private:

  QmitkMIDASMultiViewWidget *m_MIDASMultiViewWidget;
  berry::IPartListener::Pointer m_MIDASMultiViewWidgetListener;

};

#endif // QMITKMIDASBASEFUNCTIONALITY_H
