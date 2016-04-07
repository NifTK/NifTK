/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef MIDASGeneralSegmentorViewEventInterface_h
#define MIDASGeneralSegmentorViewEventInterface_h

#include <itkObject.h>
#include <itkSmartPointer.h>
#include <itkObjectFactory.h>
#include <mitkOperationActor.h>

class MIDASGeneralSegmentorView;

/**
 * \class MIDASGeneralSegmentorViewEventInterface
 * \brief Interface class, simply to callback onto MIDASGeneralSegmentorView class for Undo/Redo purposes.
 * \ingroup uk_ac_ucl_cmic_midasgeneralsegmentor_internal
 */
class MIDASGeneralSegmentorViewEventInterface: public itk::Object, public mitk::OperationActor
{
public:
  typedef MIDASGeneralSegmentorViewEventInterface       Self;
  typedef itk::SmartPointer<const Self>                 ConstPointer;
  typedef itk::SmartPointer<Self>                       Pointer;

  /// \brief Creates the object via the ITK object factory.
  itkNewMacro(Self);

  /// \brief Sets the view to callback on to.
  void SetMIDASGeneralSegmentorView( MIDASGeneralSegmentorView* view );

  /// \brief Main execution function.
  virtual void  ExecuteOperation(mitk::Operation* op);

protected:
  MIDASGeneralSegmentorViewEventInterface();
  ~MIDASGeneralSegmentorViewEventInterface();
private:
  MIDASGeneralSegmentorView* m_View;
};

#endif
