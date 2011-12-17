/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-18 09:05:48 +0000 (Fri, 18 Nov 2011) $
 Revision          : $Revision: 7804 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef QMITKCMICBASEFUNCTIONALITY_H
#define QMITKCMICBASEFUNCTIONALITY_H

#include <QmitkFunctionality.h>
#include <uk_ac_ucl_cmic_gui_qt_common_Export.h>

class QmitkStdMultiWidget;

/**
 * \class QmitkCMICBaseFunctionality
 * \brief Base view component for CMIC's plugins that require a typical ortho-viewer layout.
 * \ingroup uk_ac_ucl_cmic_gui_qt_common
 */
class CMIC_QT_COMMON QmitkCMICBaseFunctionality : public QmitkFunctionality
{

  Q_OBJECT

public:

  QmitkCMICBaseFunctionality();
  QmitkCMICBaseFunctionality(const QmitkCMICBaseFunctionality& other);
  virtual ~QmitkCMICBaseFunctionality();

  /// \brief QmitkFunctionality's activate.
  virtual void Activated();

  /// \brief QmitkFunctionality's deactivate.
  virtual void Deactivated();

  /// \brief QmitkFunctionality's changes regarding the QmitkStdMultiWidget.
  virtual void StdMultiWidgetNotAvailable();
  virtual void StdMultiWidgetAvailable(QmitkStdMultiWidget& stdMultiWidget);
  virtual void StdMultiWidgetClosed(QmitkStdMultiWidget& stdMultiWidget);

  /// \brief Observer to mitk::RenderingManager's RenderingManagerViewsInitializedEvent event.
  virtual void RenderingManagerReinitialized(const itk::EventObject&);

  /// \brief Observer to mitk::SliceController's SliceRotation event.
  virtual void SliceRotation(const itk::EventObject&);

protected slots:

protected:

  /// \brief Set available multiwidget. Called by framework.
  virtual void SetMultiWidget(QmitkStdMultiWidget* multiWidget);

  /// \brief The currently existing QmitkStdMultiWidget.
  QmitkStdMultiWidget *m_MultiWidget;

  /// \brief Tags for observers for QmitkStdMultiWidget.
  unsigned long m_RenderingManagerObserverTag;
  unsigned long m_SlicesRotationObserverTag1;
  unsigned long m_SlicesRotationObserverTag2;

private:

};

#endif // QMITKCMICBASEFUNCTIONALITY_H
