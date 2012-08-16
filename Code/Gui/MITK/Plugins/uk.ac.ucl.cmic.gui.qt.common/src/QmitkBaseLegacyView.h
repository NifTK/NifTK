/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date$
 Revision          : $Revision$
 Last modified by  : $Author$

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef QMITKBASELEGACYVIEW_H
#define QMITKBASELEGACYVIEW_H

#include <uk_ac_ucl_cmic_gui_qt_common_Export.h>

#include <QmitkAbstractView.h>

class QmitkMIDASMultiViewWidget;
class QmitkStdMultiWidget;

/**
 * \class QmitkBaseLegacyView
 * \brief Base view component for plugins that (unfortunately) must have access to QmitkStdMultiWidget or QmitkMIDASMultiViewWidget.
 *
 * At All Costs, you should try to avoid using it, as you are introducing hard coded links to widgets.
 * Instead you should use the mitkIRenderWindowPart, described here: http://www.mitk.org/wiki/ViewsWithoutMultiWidget
 *
 * \deprecated
 * \ingroup uk_ac_ucl_cmic_gui_qt_common
 */
class CMIC_QT_COMMON QmitkBaseLegacyView : public QmitkAbstractView
{

  Q_OBJECT

public:

  QmitkBaseLegacyView();
  QmitkBaseLegacyView(const QmitkBaseLegacyView& other);
  virtual ~QmitkBaseLegacyView();

protected:

  /**
   * \brief Does a lookup and returns a pointer to the QmitkStdMultiWidget from the editor.
   *
   * Should NOT Be Used. Think Very Carefully Before Using This.
   *
   * \deprecated
   */
  QmitkStdMultiWidget* GetActiveStdMultiWidget();

  /**
   * \brief Does a lookup and returns a pointer to the QmitkMIDASMultiViewWidget from the editor.
   *
   * Should NOT Be Used. Think Very Carefully Before Using This.
   *
   * \deprecated
   */
  QmitkMIDASMultiViewWidget* GetActiveMIDASMultiViewWidget();

private:

  /**
   * \brief Saves the MITK widget, if available, to avoid repeated lookups.
   */
  QmitkStdMultiWidget *m_MITKWidget;

  /**
   * \brief Saves the MIDAS widget, if available, to avoid repeated lookups.
   */
  QmitkMIDASMultiViewWidget *m_MIDASWidget;

};

#endif // QMITKBASELEGACYVIEW_H
