/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-14 11:37:54 +0100 (Wed, 14 Sep 2011) $
 Revision          : $Revision: 7310 $
 Last modified by  : $Author: ad $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef __itkIterationUpdateCommand_h
#define __itkIterationUpdateCommand_h

#include "NifTKConfigure.h"
#include "niftkITKWin32ExportHeader.h"

#include "itkCommand.h"
#include "itkSingleValuedNonLinearOptimizer.h"

namespace itk
{

/**
 * \class IterationUpdateCommand
 * \brief Simply prints out the registration params, so we can track registration.
 * 
 * ITK uses a Command/Observer pattern, so the standard optimizers invoke an
 * IterationEvent at each iteration, so you register this command to print out
 * the correct parameters. In practice, within our NifTK framework,
 * this may have been done for you as in itkSingleResolutionImageRegistrationBuilder.txx.
 * 
 * \sa SingleResolutionImageRegistrationBuilder
 */ 
class NIFTKITK_WINEXPORT   ITK_EXPORT IterationUpdateCommand : public Command
{
public:
  typedef  IterationUpdateCommand                 Self;
  typedef  itk::Command                           Superclass;
  typedef  itk::SmartPointer<Self>                Pointer;
  typedef  itk::SingleValuedNonLinearOptimizer    OptimizerType;
  typedef  const OptimizerType                   *OptimizerPointer;
  typedef  OptimizerType::ParametersType          ParametersType;
  typedef  OptimizerType::MeasureType             MeasureType;
  
  /** Run-time type information (and related methods).   */
  itkTypeMacro( IterationUpdateCommand, Command );
  
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
  IterationUpdateCommand();

  /** Both Execute methods call this. */
  virtual void DoExecute(const itk::Object * object, const itk::EventObject & event);
  
private:

  IterationUpdateCommand(const Self & other);    // Purposely not implemented
  const Self & operator=( const Self & );        // Purposely not implemented
  
};

} // end namespace

#endif

