/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkToolSelectorWidget_h
#define niftkToolSelectorWidget_h

#include <QWidget>
#include "ui_niftkToolSelectorWidget.h"

namespace niftk
{

/// \class ToolSelectorWidget
/// \brief Implements a widget containing the QmitkToolGUIArea and QmitkToolSelectionBox.
///
/// This widget provides SetEnabled() and GetEnabled(), which simplify the idea
/// of enabling/disabling the toolbox.
///
/// \ingroup uk_ac_ucl_cmic_gui_qt_common
class ToolSelectorWidget : public QWidget, private Ui::niftkToolSelectorWidget {

  Q_OBJECT

public:

  /**
   * \brief Constructor.
   */
  ToolSelectorWidget(QWidget *parent = 0);

  /**
   * \brief Destructor.
   */
  ~ToolSelectorWidget();

  /**
   * \brief Set the widget to be enabled / disabled.
   */
  void SetEnabled(bool enabled);

  /**
   * \brief Get the enabled status.
   */
  bool IsEnabled() const;

  /**
   * \brief Retrieves the tool manager using the micro services API.
   */
  mitk::ToolManager* GetToolManager() const;

  /// \brief SetToolManager
  /// nullptr is not allowed, a valid manager is required.
  /// \param toolManager
  void SetToolManager(mitk::ToolManager* toolManager);

  void SetDisplayedToolGroups(const QString& toolGroups);

signals:

  /**
   * \brief Connected to the ToolSelected(int) signal from the contained QmitkToolSelectionBox.
   */
  void ToolSelected(int toolId);

protected:

private:

  ToolSelectorWidget(const ToolSelectorWidget&);  // Purposefully not implemented.
  void operator=(const ToolSelectorWidget&);  // Purposefully not implemented.

};

}

#endif

