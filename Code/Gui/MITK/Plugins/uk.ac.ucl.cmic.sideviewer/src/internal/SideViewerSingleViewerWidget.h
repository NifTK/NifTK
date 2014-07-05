/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef SideViewerSingleViewerWidget_h
#define SideViewerSingleViewerWidget_h

#include <niftkSingleViewerWidget.h>

/**
 * \class SideViewerSingleViewerWidget
 * \brief Same as niftkSingleViewerWidget just with different renderer name prefix.
 * This is needed because the renderer name can only be set through the constructor,
 * but you cannot specify parameters of widget constructors in the Qt GUI resource
 * files.
 */
class SideViewerSingleViewerWidget : public niftkSingleViewerWidget
{
  Q_OBJECT

public:

  SideViewerSingleViewerWidget(QWidget* parent = 0)
  : niftkSingleViewerWidget(parent, 0, "side viewer")
  {
  }

};

#endif
