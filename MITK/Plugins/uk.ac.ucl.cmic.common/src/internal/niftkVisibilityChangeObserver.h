/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkVisibilityChangeObserver_h
#define niftkVisibilityChangeObserver_h

namespace mitk
{
class DataNode;
}

namespace niftk
{

class VisibilityChangeObserver
{
public:
  VisibilityChangeObserver();
  virtual ~VisibilityChangeObserver();

  virtual void onVisibilityChanged(const mitk::DataNode* node) = 0;

};

}

#endif
