/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QMITKMIDASNEWSEGMENTATIONDIALOG_H_
#define QMITKMIDASNEWSEGMENTATIONDIALOG_H_

#include <niftkMIDASGuiExports.h>
#include "QmitkNewSegmentationDialog.h"
#include <QColor>

/**
 * \class QmitkMIDASNewSegmentationDialog
 * \brief Derives from QmitkNewSegmentationDialog, to simply set the default colour to pure green.
 * \sa QmitkNewSegmentationDialog
 */
class NIFTKMIDASGUI_EXPORT QmitkMIDASNewSegmentationDialog : public QmitkNewSegmentationDialog
{
  Q_OBJECT

public:

  /// \brief Constructor, which sets the default button colour to that given by defaultColor.
  QmitkMIDASNewSegmentationDialog(const QColor &defaultColor, QWidget* parent = 0);
  ~QmitkMIDASNewSegmentationDialog() {}
};
#endif /*QMITKMIDASNEWSEGMENTATIONDIALOG_H_*/
