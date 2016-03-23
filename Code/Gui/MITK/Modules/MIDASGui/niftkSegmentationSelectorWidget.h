/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __niftkSegmentationSelectorWidget_h
#define __niftkSegmentationSelectorWidget_h

#include <niftkMIDASGuiExports.h>
#include <QWidget>
#include "ui_niftkMIDASImageAndSegmentationSelector.h"

/**
 * \class niftkSegmentationSelectorWidget
 * \brief Implements the widget to select a reference image, and create a new segmentation.
 * \ingroup uk_ac_ucl_cmic_gui_qt_common
 */
class NIFTKMIDASGUI_EXPORT niftkSegmentationSelectorWidget : public QWidget, public Ui::niftkMIDASImageAndSegmentationSelector {

  Q_OBJECT

public:

  /** Constructor. */
  niftkSegmentationSelectorWidget(QWidget *parent = 0);

  /** Destructor. */
  virtual ~niftkSegmentationSelectorWidget();

protected:

private:

  niftkSegmentationSelectorWidget(const niftkSegmentationSelectorWidget&);  // Purposefully not implemented.
  void operator=(const niftkSegmentationSelectorWidget&);  // Purposefully not implemented.

};

#endif

