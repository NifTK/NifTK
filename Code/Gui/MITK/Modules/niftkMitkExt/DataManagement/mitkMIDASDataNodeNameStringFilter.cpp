/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2012-07-17 12:27:28 +0100 (Tue, 17 Jul 2012) $
 Revision          : $Revision: 9362 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "mitkMIDASDataNodeNameStringFilter.h"
#include "mitkMIDASTool.h"
#include "mitkMIDASPolyTool.h"

namespace mitk
{

//-----------------------------------------------------------------------------
MIDASDataNodeNameStringFilter::MIDASDataNodeNameStringFilter()
{
  this->AddToList("FeedbackContourTool");
  this->AddToList("MIDASContourTool");
  this->AddToList(mitk::MIDASTool::SEED_POINT_SET_NAME);
  this->AddToList(mitk::MIDASTool::CURRENT_CONTOURS_NAME);
  this->AddToList(mitk::MIDASTool::REGION_GROWING_IMAGE_NAME);
  this->AddToList(mitk::MIDASTool::PRIOR_CONTOURS_NAME);
  this->AddToList(mitk::MIDASTool::NEXT_CONTOURS_NAME);
  this->AddToList(mitk::MIDASTool::NEXT_CONTOURS_NAME);
  this->AddToList(mitk::MIDASTool::MORPH_EDITS_EROSIONS_SUBTRACTIONS);
  this->AddToList(mitk::MIDASTool::MORPH_EDITS_EROSIONS_ADDITIONS);
  this->AddToList(mitk::MIDASTool::MORPH_EDITS_DILATIONS_SUBTRACTIONS);
  this->AddToList(mitk::MIDASTool::MORPH_EDITS_DILATIONS_ADDITIONS);
  this->AddToList(mitk::MIDASPolyTool::MIDAS_POLY_TOOL_ANCHOR_POINTS);
  this->AddToList(mitk::MIDASPolyTool::MIDAS_POLY_TOOL_PREVIOUS_CONTOUR);
  this->AddToList("Paintbrush_Node");
  this->SetPropertyName("name");
}


//-----------------------------------------------------------------------------
MIDASDataNodeNameStringFilter::~MIDASDataNodeNameStringFilter()
{

}

} // end namespace
