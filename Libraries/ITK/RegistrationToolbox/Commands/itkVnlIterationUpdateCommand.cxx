/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkVnlIterationUpdateCommand_cxx
#define __itkVnlIterationUpdateCommand_cxx

#include <sstream>
#include "itkVnlIterationUpdateCommand.h"

#include <itkUCLMacro.h>

namespace itk
{

/**
 * Constructor.
 */
VnlIterationUpdateCommand::VnlIterationUpdateCommand()
{
  niftkitkDebugMacro("VnlIterationUpdateCommand():Constructed");
}
	
void VnlIterationUpdateCommand::DoExecute(const itk::Object * object, const itk::EventObject & event)
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

  // To track registraiton, we used the cached value.
  // When the VNL optimizer exits, it sets the normal
  // values so GetCurrentValue and GetCurrentPosition will work.
  ParametersType parameters = optimizer->GetCachedCurrentPosition(); 
  MeasureType measure = optimizer->GetCachedValue();
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
