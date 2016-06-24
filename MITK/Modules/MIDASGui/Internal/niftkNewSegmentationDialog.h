/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkNewSegmentationDialog_h
#define niftkNewSegmentationDialog_h

#include <QColor>

#include <QmitkNewSegmentationDialog.h>

namespace niftk
{

/// \class NewSegmentationDialog
/// \brief Derives from QmitkNewSegmentationDialog, to simply set the default colour to pure green.
/// \sa QmitkNewSegmentationDialog
class NewSegmentationDialog : public QmitkNewSegmentationDialog
{
  Q_OBJECT

public:

  /// \brief Constructor, which sets the default button colour to that given by defaultColor.
  NewSegmentationDialog(const QColor& defaultColor, QWidget* parent = 0);

  /// \brief Destructs the NewSegmentationDialog object.
  virtual ~NewSegmentationDialog();

};

}

#endif
