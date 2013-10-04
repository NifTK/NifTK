/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkMIDASDataNodeNameStringFilter.h"
#include <mitkMIDASTool.h>
#include <mitkMIDASPolyTool.h>

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
  this->AddToList(mitk::MIDASTool::DRAW_CONTOURS_NAME);
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
