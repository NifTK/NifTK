/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkMIDASSingleViewWidgetListVisibilityManager_h
#define QmitkMIDASSingleViewWidgetListVisibilityManager_h

#include <niftkMIDASGuiExports.h>
#include "QmitkMIDASSingleViewWidgetListManager.h"
#include <vector>
#include <mitkDataStorage.h>

/**
 * \class QmitkMIDASSingleViewWidgetListVisibilityManager
 * \brief Maintains a list of QmitkMIDASSingleViewWidget and coordinates visibility properties.
 */
class NIFTKMIDASGUI_EXPORT QmitkMIDASSingleViewWidgetListVisibilityManager : public QmitkMIDASSingleViewWidgetListManager
{

public:

  /// \brief Constructor.
  QmitkMIDASSingleViewWidgetListVisibilityManager();

  /// \brief Destructor.
  virtual ~QmitkMIDASSingleViewWidgetListVisibilityManager();

  /// \brief Set the data storage.
  void SetDataStorage(mitk::DataStorage::Pointer dataStorage);

  /// \brief Sets the node to have a renderer specific visibility.
  void SetNodeVisibilityForWindow(mitk::DataNode* node, const unsigned int& widgetIndex, const bool& visibility);

  /// \brief Will query the DataStorage for all valid nodes, and for the given window,
  /// will set a renderer specific property equal to visibility.
  void SetAllNodeVisibilityForWindow(const unsigned int& widgetIndex, const bool& visibility);

  /// \brief For all currently registered windows, will make sure the node has a renderer
  /// specific visibility property equal to visibility.
  void SetNodeVisibilityForAllWindows(mitk::DataNode* node, const bool& visibility);

  /// \brief Will query the DataStorage for all valid nodes, and for all currently registered windows,
  /// will set a renderer specific property equal to visibility.
  void SetAllNodeVisibilityForAllWindows(const bool& visibility);

  /// \brief Used to clear a single window.
  void ClearWindow(const unsigned int& windowIndex);

  /// \brief Used to clear a range of windows, meaning to set renderer specific visibility
  /// properties to false for all the nodes registered with the contained list of widgets.
  void ClearWindows(const unsigned int& startWindowIndex, const unsigned int& endWindowIndex);

  /// \brief Clears all windows, meaning to set renderer specific visibility properties to
  /// false for all the nodes registered with the contained list of widgets.
  void ClearAllWindows();

private:

  mitk::DataStorage::Pointer m_DataStorage;
};

#endif
