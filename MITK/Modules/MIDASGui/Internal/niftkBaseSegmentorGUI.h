/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkBaseSegmentorGUI_h
#define niftkBaseSegmentorGUI_h

#include <niftkBaseGUI.h>

namespace mitk
{
class ToolManager;
}

namespace niftk
{

class SegmentationSelectorWidget;
class ToolSelectorWidget;

/// \class BaseSegmentorGUI
/// \brief Base class for GUI controls on MIDAS segmentor views.
///
/// \sa niftkMorphologicalSegmentorGUI
/// \sa niftkGeneralSegmentorGUI
class BaseSegmentorGUI : public BaseGUI
{
  Q_OBJECT

public:

  BaseSegmentorGUI(QWidget* parent);
  virtual ~BaseSegmentorGUI();

public:

  /// \brief Tells if the tool selection box is on/off
  bool IsToolSelectorEnabled() const;

  /// \brief Turns the tool selection box on/off
  void SetToolSelectorEnabled(bool enabled);

  /// \brief Sets the tool manager of the tool selection box.
  void SetToolManager(mitk::ToolManager* toolManager);

  /// \brief Gets the tool manager of the tool selection box.
  mitk::ToolManager* GetToolManager() const;

signals:

  /// \brief Emitted when the user clicks on the 'Start/restart segmentation' button.
  void NewSegmentationButtonClicked();

protected:

  /// \brief Method to enable derived classes to turn widgets off/on, with default do nothing implementation.
  virtual void EnableSegmentationWidgets(bool enabled);

private:

  void OnWorkingDataChanged();

protected:

  /// \brief Common widget, enabling selection of Image and Segmentation, that might be replaced once we have a database.
  SegmentationSelectorWidget *m_SegmentationSelectorWidget;

  /// \brief Common widget, enabling selection of a segmentation tool.
  ToolSelectorWidget *m_ToolSelectorWidget;

  /// \brief Container for Selector Widget.
  QWidget *m_ContainerForSelectorWidget;

  /// \brief Container for Tool Widget.
  QWidget *m_ContainerForToolWidget;

  mitk::ToolManager* m_ToolManager;

};

}

#endif
