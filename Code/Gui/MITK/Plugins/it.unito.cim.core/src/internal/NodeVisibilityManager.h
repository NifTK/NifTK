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

#ifndef __NodeVisibilityManager_h
#define __NodeVisibilityManager_h

#include "PluginCore.h"

class NodeVisibilityManagerPrivate;

namespace mitk {
class DataNode;
}

class NodeVisibilityManager : public PluginCore
{
public:

  explicit NodeVisibilityManager();
  virtual ~NodeVisibilityManager();

  virtual void onNodeAdded(const mitk::DataNode* node);
  virtual void onNodeRemoved(const mitk::DataNode* node);
  virtual void onVisibilityChanged(const mitk::DataNode* node);

private:
  void propagateVisibilityOn(const mitk::DataNode* node);
  void propagateVisibilityOff(const mitk::DataNode* node);

  QScopedPointer<NodeVisibilityManagerPrivate> d_ptr;

  Q_DECLARE_PRIVATE(NodeVisibilityManager);
  Q_DISABLE_COPY(NodeVisibilityManager);
};

#endif
