/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef VISIBILITYCHANGEOBSERVER_H_
#define VISIBILITYCHANGEOBSERVER_H_

namespace mitk {
class DataNode;
}

class VisibilityChangeObserver
{
public:
  VisibilityChangeObserver();
  virtual ~VisibilityChangeObserver();

  virtual void onVisibilityChanged(const mitk::DataNode* node) = 0;

};

#endif /* VISIBILITYCHANGEOBSERVER_H_ */
