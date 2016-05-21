/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkContourToolEventInterface_h
#define niftkContourToolEventInterface_h

#include <itkObject.h>
#include <itkObjectFactory.h>
#include <itkSmartPointer.h>

#include <mitkOperationActor.h>

namespace niftk
{

class MIDASContourTool;

/**
 * \class MIDASContourToolEventInterface
 * \brief Interface class, simply to callback onto MIDASContourTool for Undo/Redo purposes.
 */
class MIDASContourToolEventInterface: public itk::Object, public mitk::OperationActor
{
public:
  typedef MIDASContourToolEventInterface       Self;
  typedef itk::SmartPointer<const Self>        ConstPointer;
  typedef itk::SmartPointer<Self>              Pointer;

  /// \brief Creates the object via the ITK object factory.
  itkNewMacro(Self);

  /// \brief Sets the tool to callback on to.
  void SetMIDASContourTool( MIDASContourTool* tool );

  /// \brief Main execution function.
  virtual void  ExecuteOperation(mitk::Operation* op);

protected:
  MIDASContourToolEventInterface();
  ~MIDASContourToolEventInterface();
private:
  MIDASContourTool* m_Tool;
};

}

#endif
