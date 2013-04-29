/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QMITKBASELEGACYVIEW_H
#define QMITKBASELEGACYVIEW_H

#include <uk_ac_ucl_cmic_gui_qt_commonlegacy_Export.h>

#include <QmitkAbstractView.h>

class QmitkStdMultiWidget;

/**
 * \class QmitkBaseLegacyView
 * \brief Base view component for plugins that (unfortunately) must have access to QmitkStdMultiWidget.
 *
 * At All Costs, you should try to avoid using it, as you are introducing hard coded links to widgets.
 * Instead you should use the mitkIRenderWindowPart, described here: http://www.mitk.org/wiki/ViewsWithoutMultiWidget
 *
 * \deprecated
 * \ingroup uk_ac_ucl_cmic_gui_qt_commonlegacy
 */
class CMIC_QT_COMMONLEGACY QmitkBaseLegacyView : public QmitkAbstractView
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

private:

  /**
   * \brief Saves the MITK widget, if available, to avoid repeated lookups.
   */
  QmitkStdMultiWidget *m_MITKWidget;

};

#endif // QMITKBASELEGACYVIEW_H
