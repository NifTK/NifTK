/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-21 08:47:19 +0100 (Wed, 21 Sep 2011) $
 Revision          : $Revision: 7343 $
 Last modified by  : $Author: mjc $

 Original author   : j.hipwell@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef __itkReconstructionAndRegistrationUpdateCommand_cxx
#define __itkReconstructionAndRegistrationUpdateCommand_cxx

#include <sstream>
#include "itkReconstructionAndRegistrationUpdateCommand.h"
#include "itkUCLMacro.h"


namespace itk
{

/* -----------------------------------------------------------------------
   Constructor
   ----------------------------------------------------------------------- */

ReconstructionAndRegistrationUpdateCommand::ReconstructionAndRegistrationUpdateCommand()
{
  niftkitkInfoMacro(<< "ReconstructionAndRegistrationUpdateCommand():Constructed");

  m_Iteration = 0;
}
	

/* -----------------------------------------------------------------------
   Execute(itk::Object *caller, const itk::EventObject & event)
   ----------------------------------------------------------------------- */

void ReconstructionAndRegistrationUpdateCommand::Execute(itk::Object *caller, const itk::EventObject & event)
{
  if( ! itk::IterationEvent().CheckEvent( &event ) )
    return;

  this->DoExecute( const_cast<itk::Object *>(caller), event);
}


/* -----------------------------------------------------------------------
   Execute(const itk::Object * object, const itk::EventObject & event)
   ----------------------------------------------------------------------- */

void ReconstructionAndRegistrationUpdateCommand::Execute(const itk::Object * object, const itk::EventObject & event)
{
  if( ! itk::IterationEvent().CheckEvent( &event ) )
    return;

  this->DoExecute(object, event);
}


/* -----------------------------------------------------------------------
   DoExecute(const itk::Object * object, const itk::EventObject & event)
   ----------------------------------------------------------------------- */

void ReconstructionAndRegistrationUpdateCommand::DoExecute(const itk::Object * object, const itk::EventObject & /* event */)
{
  OptimizerPointer optimizer;

  try {
    optimizer = dynamic_cast< OptimizerPointer >( object );
  }

  catch( std::exception & err ) {

	niftkitkExceptionMacro("Failed to dynamic_cast optimizer");
    throw err;
  }

  if (optimizer == 0) {
	niftkitkExceptionMacro("Failed to cast optimizer, pointer is null");
  }

  std::cout << "Iteration " << ++m_Iteration << std::endl;
  //optimizer->Print(std::cout);
}

} // end namespace

#endif
