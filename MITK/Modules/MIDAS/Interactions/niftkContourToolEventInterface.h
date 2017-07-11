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

class ContourTool;

/**
 * \class ContourToolEventInterface
 * \brief Interface class, simply to callback onto ContourTool for Undo/Redo purposes.
 */
class ContourToolEventInterface: public itk::Object, public mitk::OperationActor
{
public:
  typedef ContourToolEventInterface       Self;
  typedef itk::SmartPointer<const Self>        ConstPointer;
  typedef itk::SmartPointer<Self>              Pointer;

  /// \brief Creates the object via the ITK object factory.
  itkNewMacro(Self)

  /// \brief Sets the tool to callback on to.
  void SetContourTool(ContourTool* tool);

  /// \brief Main execution function.
  virtual void  ExecuteOperation(mitk::Operation* op) override;

protected:
  ContourToolEventInterface();
  ~ContourToolEventInterface();
private:
  ContourTool* m_Tool;
};

}

#endif
