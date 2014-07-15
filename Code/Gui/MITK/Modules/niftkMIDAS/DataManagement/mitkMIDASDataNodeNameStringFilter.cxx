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
#include <mitkMIDASPaintbrushTool.h>
#include <mitkMIDASPolyTool.h>

namespace mitk
{

//-----------------------------------------------------------------------------
MIDASDataNodeNameStringFilter::MIDASDataNodeNameStringFilter()
{
  this->AddToList("One of FeedbackContourTool's feedback nodes");
  this->AddToList("MIDASContourTool");
  this->AddToList(mitk::MIDASTool::SEEDS_NAME);
  this->AddToList(mitk::MIDASTool::CONTOURS_NAME);
  this->AddToList(mitk::MIDASTool::REGION_GROWING_NAME);
  this->AddToList(mitk::MIDASTool::PRIOR_CONTOURS_NAME);
  this->AddToList(mitk::MIDASTool::NEXT_CONTOURS_NAME);
  this->AddToList(mitk::MIDASTool::DRAW_CONTOURS_NAME);
  this->AddToList(mitk::MIDASPaintbrushTool::EROSIONS_SUBTRACTIONS_NAME);
  this->AddToList(mitk::MIDASPaintbrushTool::EROSIONS_ADDITIONS_NAME);
  this->AddToList(mitk::MIDASPaintbrushTool::DILATIONS_SUBTRACTIONS_NAME);
  this->AddToList(mitk::MIDASPaintbrushTool::DILATIONS_ADDITIONS_NAME);
  this->AddToList(mitk::MIDASPolyTool::MIDAS_POLY_TOOL_ANCHOR_POINTS);
  this->AddToList(mitk::MIDASPolyTool::MIDAS_POLY_TOOL_PREVIOUS_CONTOUR);
  this->AddToList("Paintbrush_Node");
  this->AddToList("widget1Plane");
  this->AddToList("widget2Plane");
  this->AddToList("widget3Plane");
  this->SetPropertyName("name");
}


//-----------------------------------------------------------------------------
MIDASDataNodeNameStringFilter::~MIDASDataNodeNameStringFilter()
{

}

} // end namespace
