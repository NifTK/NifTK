/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-12-16 09:12:58 +0000 (Fri, 16 Dec 2011) $
 Revision          : $Revision: 8039 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "mitkMIDASNodeAddedInterpolationSetter.h"
#include <mitkDataNode.h>
#include <mitkProperties.h>
#include <mitkVtkResliceInterpolationProperty.h>

namespace mitk
{

//-----------------------------------------------------------------------------
MIDASNodeAddedInterpolationSetter::MIDASNodeAddedInterpolationSetter()
: DataStorageListener()
, m_DefaultInterpolation(MIDAS_INTERPOLATION_CUBIC)
{
}


//-----------------------------------------------------------------------------
MIDASNodeAddedInterpolationSetter::MIDASNodeAddedInterpolationSetter(mitk::DataStorage::Pointer dataStorage)
: DataStorageListener(dataStorage)
, m_DefaultInterpolation(MIDAS_INTERPOLATION_CUBIC)
{
}


//-----------------------------------------------------------------------------
MIDASNodeAddedInterpolationSetter::~MIDASNodeAddedInterpolationSetter()
{
}


//-----------------------------------------------------------------------------
void MIDASNodeAddedInterpolationSetter::NodeAdded(mitk::DataNode* node)
{
  mitk::Image* image = dynamic_cast<mitk::Image*>(node->GetData());
  if (image != NULL)
  {
    bool isBinary = false;
    node->GetBoolProperty("binary", isBinary);

    if (!isBinary)
    {
      if (m_DefaultInterpolation == MIDAS_INTERPOLATION_NONE)
      {
        node->SetProperty("texture interpolation", mitk::BoolProperty::New(false));
      }
      else
      {
        node->SetProperty("texture interpolation", mitk::BoolProperty::New(true));
      }

      mitk::VtkResliceInterpolationProperty::Pointer interpolationProperty = mitk::VtkResliceInterpolationProperty::New();

      if (m_DefaultInterpolation == MIDAS_INTERPOLATION_NONE)
      {
        interpolationProperty->SetInterpolationToNearest();
      }
      else if (m_DefaultInterpolation == MIDAS_INTERPOLATION_LINEAR)
      {
        interpolationProperty->SetInterpolationToLinear();
      }
      else if (m_DefaultInterpolation == MIDAS_INTERPOLATION_CUBIC)
      {
        interpolationProperty->SetInterpolationToCubic();
      }

      node->SetProperty("reslice interpolation", interpolationProperty);

    } // end if not binary
  } // end if is an image
}

} // end namespace
