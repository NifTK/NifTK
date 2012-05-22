/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-21 08:53:21 +0100 (Wed, 21 Sep 2011) $
 Revision          : $Revision: 7344 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef MITKMIDASCONTOURTOOLEVENTINTERFACE_H
#define MITKMIDASCONTOURTOOLEVENTINTERFACE_H

#include "itkObject.h"
#include "mitkOperationActor.h"

namespace mitk {

class MIDASContourTool;

/**
 * \class MIDASContourToolEventInterface
 * \brief Interface class, simply to callback onto MIDASContourTool for Undo/Redo purposes.
 */
class MIDASContourToolEventInterface: public itk::Object, public mitk::OperationActor
{
public:
  MIDASContourToolEventInterface();
  ~MIDASContourToolEventInterface();
  void SetMIDASContourTool( MIDASContourTool* tool );
  virtual void  ExecuteOperation(mitk::Operation* op);
private:
  MIDASContourTool* m_Tool;
};

} // end namespace

#endif
