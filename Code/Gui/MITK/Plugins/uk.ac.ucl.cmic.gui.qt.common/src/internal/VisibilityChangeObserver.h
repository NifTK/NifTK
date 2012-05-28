/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-24 15:53:45 +0000 (Thu, 24 Nov 2011) $
 Revision          : $Revision: 7857 $
 Last modified by  : $Author: mjc $

 Original author   : Miklos Espak <espakm@gmail.com>

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

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
