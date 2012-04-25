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

class QmitkMIDASMultiViewWidget;
class QmitkStdMultiWidget;

/**
 * \class QmitkMIDASBaseFunctionality
 * \brief Base view component for MIDAS plugins providing access to QmitkMIDASMultiViewWidget.
 *
 * \ingroup uk_ac_ucl_cmic_gui_qt_common
 */
class CMIC_QT_COMMON QmitkMIDASBaseFunctionality : public QmitkAbstractView
{

  Q_OBJECT

public:

  QmitkMIDASBaseFunctionality();
  QmitkMIDASBaseFunctionality(const QmitkMIDASBaseFunctionality& other);
  virtual ~QmitkMIDASBaseFunctionality();

  /// \brief Derived classes must provide a method to return the view ID.
  virtual std::string GetViewID() const = 0;

protected:

  /// \brief Saves the parent of this view.
  QWidget *m_Parent;

  /// \brief Does a lookup and returns a pointer to the QmitkStdMultiWidget from the editor.
  QmitkStdMultiWidget* GetActiveStdMultiWidget();

  /// \brief Does a lookup and returns a pointer to the QmitkMIDASMultiViewWidget from the editor.
  QmitkMIDASMultiViewWidget* GetActiveMIDASMultiViewWidget();

private:

  /// \brief Saves the MITK widget, if available.
  QmitkStdMultiWidget *m_MITKWidget;

  /// \brief Saves the MIDAS widget, if available.
  QmitkMIDASMultiViewWidget *m_MIDASWidget;

};

#endif // QMITKMIDASBASEFUNCTIONALITY_H
