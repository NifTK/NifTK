/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkGeneralSegmentorEventInterface_h
#define niftkGeneralSegmentorEventInterface_h

#include <itkObject.h>
#include <itkSmartPointer.h>
#include <itkObjectFactory.h>
#include <mitkOperationActor.h>

namespace niftk
{

class GeneralSegmentorController;

/// \class GeneralSegmentorEventInterface
/// \brief Interface class, simply to callback onto niftkGeneralSegmentorController class for Undo/Redo purposes.
/// \ingroup uk_ac_ucl_cmic_midasgeneralsegmentor_internal
class GeneralSegmentorEventInterface: public itk::Object, public mitk::OperationActor
{
public:
  typedef GeneralSegmentorEventInterface       Self;
  typedef itk::SmartPointer<const Self>             ConstPointer;
  typedef itk::SmartPointer<Self>                   Pointer;

  /// \brief Creates the object via the ITK object factory.
  itkNewMacro(Self);

  /// \brief Sets the view to callback on to.
  void SetGeneralSegmentorController(GeneralSegmentorController* generalSegmentorController);

  /// \brief Main execution function.
  virtual void ExecuteOperation(mitk::Operation* op) override;

protected:

  GeneralSegmentorEventInterface();

  virtual ~GeneralSegmentorEventInterface();

private:

  GeneralSegmentorController* m_GeneralSegmentorController;

};

}

#endif
