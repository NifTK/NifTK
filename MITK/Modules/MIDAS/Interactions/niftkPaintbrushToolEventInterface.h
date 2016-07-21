/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkPaintbrushToolEventInterface_h
#define niftkPaintbrushToolEventInterface_h

#include <itkObject.h>
#include <itkObjectFactory.h>
#include <itkSmartPointer.h>

#include <mitkOperationActor.h>

namespace niftk
{

class PaintbrushTool;

/**
 * \class PaintbrushToolEventInterface
 * \brief Interface class, simply to callback operations onto the PaintbrushTool.
 */
class PaintbrushToolEventInterface: public itk::Object, public mitk::OperationActor
{
public:
  typedef PaintbrushToolEventInterface  Self;
  typedef itk::SmartPointer<const Self>      ConstPointer;
  typedef itk::SmartPointer<Self>            Pointer;

  /// \brief Creates the object via the ITK object factory.
  itkNewMacro(Self);

  /// \brief Sets the tool to callback on to.
  void SetPaintbrushTool(PaintbrushTool* paintbrushTool);

  /// \brief Main execution function.
  virtual void ExecuteOperation(mitk::Operation* op) override;

protected:
  PaintbrushToolEventInterface();
  ~PaintbrushToolEventInterface();
private:
  PaintbrushTool* m_PaintbrushTool;
};

}

#endif

