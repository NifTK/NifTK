/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkMIDASNewSegmentationDialog_h
#define niftkMIDASNewSegmentationDialog_h

#include <niftkMIDASGuiExports.h>

#include <QColor>

#include <QmitkNewSegmentationDialog.h>

/**
 * \class niftkMIDASNewSegmentationDialog
 * \brief Derives from niftkNewSegmentationDialog, to simply set the default colour to pure green.
 * \sa niftkNewSegmentationDialog
 */
class NIFTKMIDASGUI_EXPORT niftkMIDASNewSegmentationDialog : public QmitkNewSegmentationDialog
{
  Q_OBJECT

public:

  /// \brief Constructor, which sets the default button colour to that given by defaultColor.
  niftkMIDASNewSegmentationDialog(const QColor& defaultColor, QWidget* parent = 0);

  /// \brief Destructs the niftkMIDASNewSegmentationDialog object.
  virtual ~niftkMIDASNewSegmentationDialog();

};
#endif
