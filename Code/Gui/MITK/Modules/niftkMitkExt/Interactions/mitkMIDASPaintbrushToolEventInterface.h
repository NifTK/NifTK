/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date$
 Revision          : $Revision$
 Last modified by  : $Author$

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef MITKMIDASPAINTBRUSHTOOLEVENTINTERFACE_H
#define MITKMIDASPAINTBRUSHTOOLEVENTINTERFACE_H

#include "itkObject.h"
#include "mitkOperationActor.h"

namespace mitk
{

class MIDASPaintbrushTool;

/**
 * \class MIDASPaintbrushToolEventInterface
 * \brief Interface class, simply to callback operations onto the MIDASPaintbrushTool.
 */
class MIDASPaintbrushToolEventInterface: public itk::Object, public mitk::OperationActor
{
public:
  MIDASPaintbrushToolEventInterface();
  ~MIDASPaintbrushToolEventInterface();
  void SetMIDASPaintbrushTool( MIDASPaintbrushTool* paintbrushTool );
  virtual void  ExecuteOperation(mitk::Operation* op);
private:
  MIDASPaintbrushTool* m_MIDASPaintBrushTool;
};

} // end namespace

#endif

