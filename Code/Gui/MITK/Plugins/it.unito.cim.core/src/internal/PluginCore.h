/*=============================================================================

 KMaps:     An image processing toolkit for DCE-MRI analysis developed
            at the Molecular Imaging Center at University of Torino.

 See:       http://www.cim.unito.it

 Author:    Miklos Espak <espakm@gmail.com>

 Copyright (c) Miklos Espak
 All Rights Reserved.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef __PluginCore_h
#define __PluginCore_h

#include <berryIWorkbenchPart.h>

#include <mitkDataNodeSelection.h>

#include "VisibilityChangeObserver.h"

#include <vector>

class PluginCorePrivate;

namespace mitk {
class DataNode;
class DataStorage;
}

class QmitkStdMultiWidget;

class PluginCore : public VisibilityChangeObserver
{

public:

  explicit PluginCore();
  virtual ~PluginCore();

  mitk::DataStorage* GetDataStorage();

  virtual void onNodeAdded(const mitk::DataNode* node);
  virtual void onNodeRemoved(const mitk::DataNode* node);
//  virtual void onNodeChanged(const mitk::DataNode* node);

  ///
  /// Called when the selection in the workbench changed
  ///
  virtual void OnSelectionChanged(std::vector<mitk::DataNode*> /*nodes*/);

  ///
  /// Called when the visibility of a node in the data storage changed
  ///
  virtual void onVisibilityChanged(const mitk::DataNode* node);

  ///
  /// \return the selection of the currently active part of the workbench or an empty vector
  ///         if nothing is selected
  ///
  std::vector<mitk::DataNode*> GetCurrentSelection() const;
  ///
  /// Returns the current selection made in the datamanager bundle or an empty vector
  /// if nothing`s selected or if the bundle does not exist
  ///
  std::vector<mitk::DataNode*> GetDataManagerSelection() const;

protected:
  virtual void init();
//  QmitkStdMultiWidget* GetActiveStdMultiWidget();

private:
  void onNodeAddedInternal(const mitk::DataNode*);
  void onNodeRemovedInternal(const mitk::DataNode*);
//  void onVisibilityChangedInternal(const mitk::DataNode*);
  ///
  /// reactions to selection events from data manager (and potential other senders)
  ///
  void BlueBerrySelectionChanged(berry::IWorkbenchPart::Pointer sourcepart, berry::ISelection::ConstPointer selection);
  ///
  /// Converts a mitk::DataNodeSelection to a std::vector<mitk::DataNode*> (possibly empty
  ///
  std::vector<mitk::DataNode*> DataNodeSelectionToVector(mitk::DataNodeSelection::ConstPointer currentSelection) const;

  QScopedPointer<PluginCorePrivate> d_ptr;

  Q_DECLARE_PRIVATE(PluginCore);
  Q_DISABLE_COPY(PluginCore);
};

#endif
