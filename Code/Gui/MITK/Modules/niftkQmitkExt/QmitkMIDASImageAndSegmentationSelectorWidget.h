/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

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

