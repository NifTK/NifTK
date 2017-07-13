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

namespace mitk
{
class ToolManager;
}

namespace niftk
{

/// \class SegmentationSelectorWidget
/// \brief Implements the widget to select a reference image, and create a new segmentation.
///
/// The class track the selected reference and working data in the tool manager,
/// and if a valid reference image and segmentation is selected, displays their name.
/// If no image is selected, it desplays a default message ("not selected") in red.
///
class SegmentationSelectorWidget : public QWidget, private Ui::niftkSegmentationSelectorWidget
{

  Q_OBJECT

public:

  SegmentationSelectorWidget(QWidget* parent = 0);

  virtual ~SegmentationSelectorWidget();

  /// \brief Retrieves the tool manager using the micro services API.
  mitk::ToolManager* GetToolManager() const;

  /// \brief Sets the tool manager
  /// nullptr is not allowed, a valid manager is required.
  /// \param toolManager
  void SetToolManager(mitk::ToolManager* toolManager);

signals:

  void NewSegmentationButtonClicked();

private:

  /// \brief Displays the name of the reference image on a label.
  /// If there is no reference image selected then it displays the "not selected" message in red.
  void OnReferenceDataChanged();

  /// \brief Displays the name of the segmentation image on a label.
  /// If there is no segmentation image selected then it displays the "not selected" message in red.
  void OnWorkingDataChanged();

  mitk::ToolManager* m_ToolManager;

  SegmentationSelectorWidget(const SegmentationSelectorWidget&);  // Purposefully not implemented.
  void operator=(const SegmentationSelectorWidget&);  // Purposefully not implemented.

};

}

#endif
