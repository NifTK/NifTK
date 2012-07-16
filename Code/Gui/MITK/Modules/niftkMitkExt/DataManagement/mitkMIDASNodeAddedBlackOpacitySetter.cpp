/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date$
 Revision          : $Revision$
 Last modified by  : $Author$

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "mitkMIDASNodeAddedBlackOpacitySetter.h"
#include <mitkDataNode.h>
#include <mitkProperties.h>
#include <mitkVtkResliceInterpolationProperty.h>

namespace mitk
{

//-----------------------------------------------------------------------------
MIDASNodeAddedBlackOpacitySetter::MIDASNodeAddedBlackOpacitySetter()
{
}


//-----------------------------------------------------------------------------
MIDASNodeAddedBlackOpacitySetter::MIDASNodeAddedBlackOpacitySetter(mitk::DataStorage::Pointer dataStorage)
: DataStorageListener(dataStorage)
{
}


//-----------------------------------------------------------------------------
MIDASNodeAddedBlackOpacitySetter::~MIDASNodeAddedBlackOpacitySetter()
{
}


//-----------------------------------------------------------------------------
void MIDASNodeAddedBlackOpacitySetter::NodeAdded(mitk::DataNode* node)
{
  // For MIDAS, which might have a light background in the render window, we need to make sure black is not transparent.
  // See MITK bug: http://bugs.mitk.org/show_bug.cgi?id=10174 which hasn't yet been completed.
  // See also Trac ticket https://cmicdev.cs.ucl.ac.uk/trac/ticket/853 where we provide a property "black opacity".

  mitk::Image* image = dynamic_cast<mitk::Image*>(node->GetData());
  if (image != NULL)
  {
    node->SetProperty("black opacity", mitk::FloatProperty::New(1));
  } // end if is an image
}

} // end namespace
