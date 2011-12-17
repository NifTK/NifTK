/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-20 14:34:44 +0100 (Tue, 20 Sep 2011) $
 Revision          : $Revision: 7333 $
 Last modified by  : $Author: ad $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef __itkVnlIterationUpdateCommand_cxx
#define __itkVnlIterationUpdateCommand_cxx

#include <sstream>
#include "itkVnlIterationUpdateCommand.h"

#include "itkUCLMacro.h"

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
