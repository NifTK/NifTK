/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkReconstructionAndRegistrationUpdateCommand_cxx
#define __itkReconstructionAndRegistrationUpdateCommand_cxx

#include <sstream>
#include "itkReconstructionAndRegistrationUpdateCommand.h"
#include <itkUCLMacro.h>


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
