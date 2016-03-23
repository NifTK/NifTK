/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __niftkBaseSegmentationViewControls_h
#define __niftkBaseSegmentationViewControls_h

#include <uk_ac_ucl_cmic_gui_qt_commonmidas_Export.h>

// CTK for event handling.
#include <service/event/ctkEventHandler.h>
#include <service/event/ctkEventAdmin.h>

// Berry stuff for application framework.
#include <berryIPreferences.h>
#include <berryIPreferencesService.h>
#include <berryIBerryPreferences.h>

// Qmitk for Qt/MITK stuff.
#include <QmitkBaseView.h>
#include <niftkMIDASImageAndSegmentationSelectorWidget.h>
#include <niftkMIDASToolSelectorWidget.h>
#include <niftkMIDASOrientationUtils.h>

// Miscellaneous.
#include <mitkToolManager.h>
#include <itkImage.h>

#include <niftkMIDASEventFilter.h>

class QmitkRenderWindow;

/**
 * \class niftkBaseSegmentationView
 * \brief Base view component for MIDAS Segmentation widgets.
 *
 * \ingroup uk_ac_ucl_cmic_gui_qt_common
 *
 * \sa QmitkBaseView
 * \sa MIDASMorphologicalSegmentorView
 * \sa MIDASGeneralSegmentorView
 * \sa MITKSegmentationView
 */
class CMIC_QT_COMMONMIDAS niftkBaseSegmentationViewControls : public QObject
{

  Q_OBJECT

public:

  niftkBaseSegmentationViewControls(QWidget* parent);
  virtual ~niftkBaseSegmentationViewControls();

public:

  /// \brief Method to enable derived classes to turn widgets off/on, with default do nothing implementation.
  virtual void EnableSegmentationWidgets(bool enabled);

  /// \brief Turns the tool selection box on/off
  virtual void SetEnableManualToolSelectionBox(bool enabled);

  /// \brief Creates the Qt connections.
  virtual void CreateConnections();

  /// \brief Sets the tool manager of the tool selection box.
  void SetToolManager(mitk::ToolManager* toolManager);

  /// \brief Gets the tool manager of the tool selection box.
  mitk::ToolManager* GetToolManager() const;

  /// \brief Common widget, enabling selection of Image and Segmentation, that might be replaced once we have a database.
  niftkMIDASImageAndSegmentationSelectorWidget *m_ImageAndSegmentationSelector;

  /// \brief Common widget, enabling selection of a segmentation tool.
  niftkMIDASToolSelectorWidget *m_ToolSelector;

  /// \brief Container for Selector Widget.
  QWidget *m_ContainerForSelectorWidget;

  /// \brief Container for Tool Widget.
  QWidget *m_ContainerForToolWidget;

};

#endif
