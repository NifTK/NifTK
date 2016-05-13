/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkMIDASDataNodeNameStringFilter.h"
#include <niftkMIDASTool.h>
#include <niftkMIDASPaintbrushTool.h>
#include <niftkMorphologicalSegmentorPipelineManager.h>
#include <niftkMIDASPolyTool.h>

namespace niftk
{

//-----------------------------------------------------------------------------
MIDASDataNodeNameStringFilter::MIDASDataNodeNameStringFilter()
{
  this->AddToList("One of FeedbackContourTool's feedback nodes");
  this->AddToList("MIDASContourTool");
  this->AddToList(MIDASTool::SEEDS_NAME);
  this->AddToList(MIDASTool::CONTOURS_NAME);
  this->AddToList(MIDASTool::REGION_GROWING_NAME);
  this->AddToList(MIDASTool::PRIOR_CONTOURS_NAME);
  this->AddToList(MIDASTool::NEXT_CONTOURS_NAME);
  this->AddToList(MIDASTool::DRAW_CONTOURS_NAME);
  this->AddToList(MIDASPaintbrushTool::EROSIONS_SUBTRACTIONS_NAME);
  this->AddToList(MIDASPaintbrushTool::EROSIONS_ADDITIONS_NAME);
  this->AddToList(MIDASPaintbrushTool::DILATIONS_SUBTRACTIONS_NAME);
  this->AddToList(MIDASPaintbrushTool::DILATIONS_ADDITIONS_NAME);
  this->AddToList(MorphologicalSegmentorPipelineManager::SEGMENTATION_OF_LAST_STAGE_NAME);
  this->AddToList(MIDASPolyTool::MIDAS_POLY_TOOL_ANCHOR_POINTS);
  this->AddToList(MIDASPolyTool::MIDAS_POLY_TOOL_PREVIOUS_CONTOUR);
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

}
