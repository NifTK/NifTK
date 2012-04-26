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
