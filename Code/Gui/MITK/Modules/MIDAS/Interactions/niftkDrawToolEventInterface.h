/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkDrawToolEventInterface_h
#define niftkDrawToolEventInterface_h

#include <itkObject.h>
#include <itkObjectFactory.h>
#include <itkSmartPointer.h>

#include <mitkOperationActor.h>

namespace niftk
{

class DrawTool;

/**
 * \class DrawToolEventInterface
 * \brief Interface class, simply to callback onto DrawTool for Undo/Redo purposes.
 */
class DrawToolEventInterface: public itk::Object, public mitk::OperationActor
{
public:
  typedef DrawToolEventInterface       Self;
  typedef itk::SmartPointer<const Self>     ConstPointer;
  typedef itk::SmartPointer<Self>           Pointer;

  /// \brief Creates the object via the ITK object factory.
  itkNewMacro(Self);

  /// \brief Sets the tool to callback on to.
  void SetDrawTool( DrawTool* tool );

  /// \brief Main execution function.
  virtual void  ExecuteOperation(mitk::Operation* op) override;

protected:
  DrawToolEventInterface();
  ~DrawToolEventInterface();
private:
  DrawTool* m_Tool;
};

}

#endif
