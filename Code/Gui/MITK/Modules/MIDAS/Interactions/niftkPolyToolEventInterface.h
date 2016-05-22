/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkPolyToolEventInterface_h
#define niftkPolyToolEventInterface_h

#include <itkObject.h>
#include <itkObjectFactory.h>
#include <itkSmartPointer.h>

#include <mitkOperationActor.h>

namespace niftk
{

class PolyTool;

/**
 * \class PolyToolEventInterface
 * \brief Interface class, simply to callback onto PolyTool for Undo/Redo purposes.
 */
class PolyToolEventInterface: public itk::Object, public mitk::OperationActor
{
public:
  typedef PolyToolEventInterface       Self;
  typedef itk::SmartPointer<const Self>     ConstPointer;
  typedef itk::SmartPointer<Self>           Pointer;

  /// \brief Creates the object via the ITK object factory.
  itkNewMacro(Self);

  /// \brief Sets the tool to callback on to.
  void SetPolyTool( PolyTool* tool );

  /// \brief Main execution function.
  virtual void ExecuteOperation(mitk::Operation* op) override;

protected:
  PolyToolEventInterface();
  ~PolyToolEventInterface();
private:
  PolyTool* m_Tool;
};

}

#endif
