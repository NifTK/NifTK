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
#ifndef MITKMIDASPOLYTOOLEVENTINTERFACE_H
#define MITKMIDASPOLYTOOLEVENTINTERFACE_H

#include "itkObject.h"
#include "itkSmartPointer.h"
#include "itkObjectFactory.h"
#include "mitkOperationActor.h"

namespace mitk {

class MIDASPolyTool;

/**
 * \class MIDASPolyToolEventInterface
 * \brief Interface class, simply to callback onto MIDASPolyTool for Undo/Redo purposes.
 */
class MIDASPolyToolEventInterface: public itk::Object, public mitk::OperationActor
{
public:
  typedef MIDASPolyToolEventInterface       Self;
  typedef itk::SmartPointer<const Self>     ConstPointer;
  typedef itk::SmartPointer<Self>           Pointer;

  /// \brief Creates the object via the ITK object factory.
  itkNewMacro(Self);

  /// \brief Sets the tool to callback on to.
  void SetMIDASPolyTool( MIDASPolyTool* tool );

  /// \brief Main execution function.
  virtual void  ExecuteOperation(mitk::Operation* op);

protected:
  MIDASPolyToolEventInterface();
  ~MIDASPolyToolEventInterface();
private:
  MIDASPolyTool* m_Tool;
};

} // end namespace

#endif
