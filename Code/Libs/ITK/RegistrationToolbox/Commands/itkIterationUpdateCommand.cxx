/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkIterationUpdateCommand_cxx
#define __itkIterationUpdateCommand_cxx

#include <sstream>
#include "itkIterationUpdateCommand.h"

#include "itkUCLMacro.h"

namespace itk
{
/**
 * Constructor.
 */
IterationUpdateCommand::IterationUpdateCommand()
{
  niftkitkDebugMacro("IterationUpdateCommand():Constructed");
}
	
void IterationUpdateCommand::Execute(itk::Object *caller, const itk::EventObject & event)
{
  this->DoExecute( const_cast<itk::Object *>(caller), event);
}

void IterationUpdateCommand::Execute(const itk::Object * object, const itk::EventObject & event)
{
  this->DoExecute(object, event);
}

void IterationUpdateCommand::DoExecute(const itk::Object * object, const itk::EventObject & event)
{
  if( ! itk::IterationEvent().CheckEvent( &event ) )
    {
      return;
    }

  OptimizerPointer optimizer;
  try
    {
      optimizer = dynamic_cast< OptimizerPointer >( object );
    }
  catch( std::exception & err )
    {
      std::string msg = "Failed to dynamic_cast optimizer";
      niftkitkErrorMacro(<< msg);
      throw err;
    }

  if (optimizer == 0)
    {
      std::string msg = "Failed to cast optimizer, pointer is null";
      niftkitkErrorMacro(<< msg);
    }

  ParametersType parameters = optimizer->GetCurrentPosition(); 
  
  // Does this recompute the cost function?
  // That depends on the derived class. The ones I checked were cached.
  MeasureType measure = optimizer->GetValue(parameters);
  if (parameters.GetSize() > 20)
    {
	  niftkitkInfoMacro(<<"DoExecute():" << measure );
    }
  else
    {
	  niftkitkInfoMacro(<<"DoExecute():" << measure << " : " << parameters);
    }
}

} // end namespace

#endif
