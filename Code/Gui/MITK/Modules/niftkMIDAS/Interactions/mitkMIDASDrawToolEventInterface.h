/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef MITKMIDASDRAWTOOLEVENTINTERFACE_H
#define MITKMIDASDRAWTOOLEVENTINTERFACE_H

#include <itkObject.h>
#include <itkSmartPointer.h>
#include <itkObjectFactory.h>
#include <mitkOperationActor.h>

namespace mitk {

class MIDASDrawTool;

/**
 * \class MIDASDrawToolEventInterface
 * \brief Interface class, simply to callback onto MIDASDrawTool for Undo/Redo purposes.
 */
class MIDASDrawToolEventInterface: public itk::Object, public mitk::OperationActor
{
public:
  typedef MIDASDrawToolEventInterface       Self;
  typedef itk::SmartPointer<const Self>     ConstPointer;
  typedef itk::SmartPointer<Self>           Pointer;

  /// \brief Creates the object via the ITK object factory.
  itkNewMacro(Self);

  /// \brief Sets the tool to callback on to.
  void SetMIDASDrawTool( MIDASDrawTool* tool );

  /// \brief Main execution function.
  virtual void  ExecuteOperation(mitk::Operation* op);

protected:
  MIDASDrawToolEventInterface();
  ~MIDASDrawToolEventInterface();
private:
  MIDASDrawTool* m_Tool;
};

} // end namespace

#endif
