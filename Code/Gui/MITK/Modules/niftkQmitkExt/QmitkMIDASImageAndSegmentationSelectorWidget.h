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

#ifndef QMITKMIDASIMAGEANDSEGMENTATIONSELECTORWIDGET_H
#define QMITKMIDASIMAGEANDSEGMENTATIONSELECTORWIDGET_H

#include <niftkQmitkExtExports.h>
#include <QWidget>
#include "ui_QmitkMIDASImageAndSegmentationSelector.h"

/**
 * \class QmitkMIDASImageAndSegmentationSelectorWidget
 * \brief Implements the widget to select a reference image, and create a new segmentation.
 * \ingroup uk_ac_ucl_cmic_gui_qt_common
 */
class NIFTKQMITKEXT_EXPORT QmitkMIDASImageAndSegmentationSelectorWidget : public QWidget, public Ui::QmitkMIDASImageAndSegmentationSelector {

  Q_OBJECT

public:

  /** Constructor. */
  QmitkMIDASImageAndSegmentationSelectorWidget(QWidget *parent = 0);

  /** Destructor. */
  ~QmitkMIDASImageAndSegmentationSelectorWidget();

protected:

private:

  QmitkMIDASImageAndSegmentationSelectorWidget(const QmitkMIDASImageAndSegmentationSelectorWidget&);  // Purposefully not implemented.
  void operator=(const QmitkMIDASImageAndSegmentationSelectorWidget&);  // Purposefully not implemented.

};

#endif

