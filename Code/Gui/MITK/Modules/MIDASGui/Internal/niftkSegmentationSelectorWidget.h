/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkSegmentationSelectorWidget_h
#define niftkSegmentationSelectorWidget_h

#include <QWidget>
#include "ui_niftkSegmentationSelectorWidget.h"

namespace niftk
{

/// \class SegmentationSelectorWidget
/// \brief Implements the widget to select a reference image, and create a new segmentation.
class SegmentationSelectorWidget : public QWidget, private Ui::niftkSegmentationSelectorWidget
{

  Q_OBJECT

public:

  SegmentationSelectorWidget(QWidget *parent = 0);

  virtual ~SegmentationSelectorWidget();

  /// \brief Displays the name of the reference image on a label.
  /// If no argument or empty string is passed then it displays the "not selected" message in red.
  void SelectReferenceImage(const QString& referenceImage = QString::null);

  /// \brief Displays the name of the segmentation image on a label.
  /// If no argument or empty string is passed then it displays the "not selected" message in red.
  void SelectSegmentationImage(const QString& segmentationImage = QString::null);

signals:

  void NewSegmentationButtonClicked();

protected:

private:

  SegmentationSelectorWidget(const SegmentationSelectorWidget&);  // Purposefully not implemented.
  void operator=(const SegmentationSelectorWidget&);  // Purposefully not implemented.

};

}

#endif
