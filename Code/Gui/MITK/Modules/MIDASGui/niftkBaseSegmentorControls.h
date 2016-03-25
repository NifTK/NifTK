/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __niftkBaseSegmentorControls_h
#define __niftkBaseSegmentorControls_h

#include <niftkMIDASGuiExports.h>

#include <QObject>

namespace mitk
{
class ToolManager;
}

class niftkSegmentationSelectorWidget;
class niftkToolSelectorWidget;

/**
 * \class niftkBaseSegmentorControls
 * \brief Base class for GUI controls on MIDAS segmentor views.
 *
 * \sa niftkMorphologicalSegmentorControls
 * \sa niftkGeneralSegmentorControls
 * \sa MITKSegmentationView
 */
class NIFTKMIDASGUI_EXPORT niftkBaseSegmentorControls : public QObject
{

  Q_OBJECT

public:

  niftkBaseSegmentorControls(QWidget* parent);
  virtual ~niftkBaseSegmentorControls();

public:

  /// \brief Method to enable derived classes to turn widgets off/on, with default do nothing implementation.
  virtual void EnableSegmentationWidgets(bool enabled);

  /// \brief Turns the tool selection box on/off
  virtual void SetEnableManualToolSelectionBox(bool enabled);

  /// \brief Sets the tool manager of the tool selection box.
  void SetToolManager(mitk::ToolManager* toolManager);

  /// \brief Gets the tool manager of the tool selection box.
  mitk::ToolManager* GetToolManager() const;

  /// \brief Selects the reference image in the internal segmentation selector widget.
  /// If no argument or empty string is passed then it unselects the reference image
  /// and displays a default message.
  void SelectReferenceImage(const QString& referenceImage = QString::null);

  /// \brief Selects the segmentation image in the internal segmentation selector widget.
  /// If no argument or empty string is passed then it unselects the reference image
  /// and displays a default message.
  void SelectSegmentationImage(const QString& segmentationImage = QString::null);

signals:

  void NewSegmentationButtonClicked();

protected:

  /// \brief Common widget, enabling selection of Image and Segmentation, that might be replaced once we have a database.
  niftkSegmentationSelectorWidget *m_SegmentationSelectorWidget;

public:

  /// \brief Common widget, enabling selection of a segmentation tool.
  niftkToolSelectorWidget *m_ToolSelectorWidget;

protected:

  /// \brief Container for Selector Widget.
  QWidget *m_ContainerForSelectorWidget;

  /// \brief Container for Tool Widget.
  QWidget *m_ContainerForToolWidget;

};

#endif
