/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkReconstructionUpdateCommand_h
#define itkReconstructionUpdateCommand_h

#include <NifTKConfigure.h>
#include <niftkITKWin32ExportHeader.h>

#include <itkCommand.h>
#include <itkSingleValuedNonLinearOptimizer.h>

namespace itk
{

/**
 * \class ReconstructionUpdateCommand
 * \brief Override this class to redefine DoExecute().
 * 
 * ITK uses a Command/Observer pattern, so the standard optimizers invoke an
 * IterationEvent at each iteration, so you register this command to print out
 * the correct parameters. In practice, within our NifTK framework,
 * this may have been done for you.
 */ 
class NIFTKITK_WINEXPORT ITK_EXPORT ReconstructionUpdateCommand : public Command
{
public:
  typedef  ReconstructionUpdateCommand            Self;
  typedef  itk::Command                           Superclass;
  typedef  itk::SmartPointer<Self>                Pointer;

  typedef  itk::SingleValuedNonLinearOptimizer    OptimizerType;
  typedef  const OptimizerType                   *OptimizerPointer;
  typedef  OptimizerType::ParametersType          ParametersType;
  typedef  OptimizerType::MeasureType             MeasureType;
  
  /** Run-time type information (and related methods).   */
  itkTypeMacro( ReconstructionUpdateCommand, Command );
  
  /** New macro for creation of through a Smart Pointer   */
  itkNewMacro( Self );

 /**
  * Calls DoExecute.
  */
 void Execute(itk::Object *caller, const itk::EventObject & event);

 /**
  * Calls DoExecute.
  */
 void Execute(const itk::Object * object, const itk::EventObject & event);

protected:

  /** No parameter constructor. */
  ReconstructionUpdateCommand();

  /** Both Execute methods call this. */
  virtual void DoExecute(const itk::Object * object, const itk::EventObject & event) ;
  
private:

  ReconstructionUpdateCommand(const Self & other);    // Purposely not implemented
  const Self & operator=( const Self & );        // Purposely not implemented
  
  unsigned int m_Iteration;

};

} // end namespace

#endif

