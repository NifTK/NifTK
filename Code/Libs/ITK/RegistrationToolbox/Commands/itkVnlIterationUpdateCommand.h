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
#ifndef __itkVnlIterationUpdateCommand_h
#define __itkVnlIterationUpdateCommand_h

#include "NifTKConfigure.h"
#include "niftkITKWin32ExportHeader.h"

#include "itkIterationUpdateCommand.h"
#include "itkSingleValuedNonLinearVnlOptimizer.h"


namespace itk
{

/**
 * \class VnlIterationUpdateCommand
 * \brief Simply prints out the registration params, so we can track registration.
 * 
 * ITK uses a Command/Observer pattern, so the standard optimizers invoke an
 * IterationEvent at each iteration, so you register this command to print out
 * the correct parameters. In practice, within our NifTK framework,
 * this may have been done for you as in itkSingleResolutionImageRegistrationBuilder.txx.
 * 
 * This one is specifically just for VNL optimizers, as they use an adaptor 
 * mechanism for passing back current parameters to the ITK framework.
 * 
 * \sa SingleResolutionImageRegistrationBuilder
 */ 
class NIFTKITK_WINEXPORT   ITK_EXPORT VnlIterationUpdateCommand : public IterationUpdateCommand
{
public:
  typedef  VnlIterationUpdateCommand              Self;
  typedef  itk::IterationUpdateCommand            Superclass;
  typedef  itk::SmartPointer<Self>                Pointer;
  typedef  itk::SingleValuedNonLinearVnlOptimizer OptimizerType;
  typedef  const OptimizerType                   *OptimizerPointer;
  typedef  Superclass::ParametersType             ParametersType;
  typedef  Superclass::MeasureType                MeasureType;
  
  /** Run-time type information (and related methods).   */
  itkTypeMacro( VnlIterationUpdateCommand, IterationUpdateCommand );
  
  /** New macro for creation of through a Smart Pointer   */
  itkNewMacro( Self );

protected:

  /** No parameter constructor. */
  VnlIterationUpdateCommand();

  /** Both Execute methods call this. */
  virtual void DoExecute(const itk::Object * object, const itk::EventObject & event);
  
private:

  VnlIterationUpdateCommand(const Self & other); // Purposely not implemented
  const Self & operator=( const Self & );        // Purposely not implemented
  
};

} // end namespace

#endif

