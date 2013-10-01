/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkMIDASSingleViewWidgetListDropManager_h
#define QmitkMIDASSingleViewWidgetListDropManager_h

#include <niftkDnDDisplayExports.h>
#include <vector>
#include "QmitkMIDASSingleViewWidgetListManager.h"
#include <mitkMIDASEnums.h>
#include <mitkDataStorage.h>

namespace mitk
{
class DataNode;
}

class QmitkRenderWindow;
class QmitkMIDASSingleViewWidgetListVisibilityManager;

/**
 * \class QmitkMIDASSingleViewWidgetListDropManager
 * \brief Class to coordinate the necessary operations for when we drop images into a
 * MIDAS QmitkMIDASMultiViewWidget, coordinating across many QmitkMIDASSingleViewWidget.
 *
 * This class needs to have SetVisibilityManager and SetDataStorage called prior to use.
 */
class NIFTKDNDDISPLAY_EXPORT QmitkMIDASSingleViewWidgetListDropManager : public QmitkMIDASSingleViewWidgetListManager
{

public:

  /// \brief Constructor.
  QmitkMIDASSingleViewWidgetListDropManager();

  /// \brief Destructor.
  virtual ~QmitkMIDASSingleViewWidgetListDropManager();

  /// \brief When nodes are dropped, we set all the default properties, and renderer specific visibility flags etc.
  void OnNodesDropped(QmitkRenderWindow *window, std::vector<mitk::DataNode*> nodes);

  /// \brief Set the visibility manager for this class to use.
  void SetVisibilityManager(QmitkMIDASSingleViewWidgetListVisibilityManager*);

  /// \brief Set the data storage.
  void SetDataStorage(mitk::DataStorage::Pointer dataStorage);

  /// \brief Sets the default layout.
  void SetDefaultLayout(MIDASLayout layout);

  /// \brief Gets the default layout.
  MIDASLayout GetDefaultLayout() const;

  /// \brief Set the drop type, which controls the behaviour when multiple images are dropped into a single widget.
  void SetDropType(const MIDASDropType& dropType);

  /// \brief Get the drop type, which controls the behaviour when multiple images are dropped into a single widget.
  MIDASDropType GetDropType() const;

  /// \brief Set the flag to determine if we accumulate images to a single geometry.
  void SetAccumulateWhenDropped(const bool& accumulateWhenDropped);

  /// \brief Get the flag to determine if we accumulate images to a single geometry.
  bool GetAccumulateWhenDropped() const;

protected:

private:

  MIDASLayout m_DefaultLayout;
  MIDASDropType m_DropType;
  bool m_AccumulateWhenDropped;
  mitk::DataStorage::Pointer m_DataStorage;
  QmitkMIDASSingleViewWidgetListVisibilityManager* m_VisibilityManager;
};

#endif
