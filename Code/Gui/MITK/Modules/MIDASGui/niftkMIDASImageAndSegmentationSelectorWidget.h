/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __niftkMIDASImageAndSegmentationSelectorWidget_h
#define __niftkMIDASImageAndSegmentationSelectorWidget_h

#include <niftkMIDASGuiExports.h>
#include <QWidget>
#include "ui_niftkMIDASImageAndSegmentationSelector.h"

/**
 * \class niftkMIDASImageAndSegmentationSelectorWidget
 * \brief Implements the widget to select a reference image, and create a new segmentation.
 * \ingroup uk_ac_ucl_cmic_gui_qt_common
 */
class NIFTKMIDASGUI_EXPORT niftkMIDASImageAndSegmentationSelectorWidget : public QWidget, public Ui::niftkMIDASImageAndSegmentationSelector {

  Q_OBJECT

public:

  /** Constructor. */
  niftkMIDASImageAndSegmentationSelectorWidget(QWidget *parent = 0);

  /** Destructor. */
  virtual ~niftkMIDASImageAndSegmentationSelectorWidget();

protected:

private:

  niftkMIDASImageAndSegmentationSelectorWidget(const niftkMIDASImageAndSegmentationSelectorWidget&);  // Purposefully not implemented.
  void operator=(const niftkMIDASImageAndSegmentationSelectorWidget&);  // Purposefully not implemented.

};

#endif

