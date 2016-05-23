/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkToolWorkingDataNameFilter.h"

#include <niftkMorphologicalSegmentorPipelineManager.h>
#include <niftkPaintbrushTool.h>
#include <niftkPolyTool.h>
#include <niftkTool.h>

namespace niftk
{

//-----------------------------------------------------------------------------
ToolWorkingDataNameFilter::ToolWorkingDataNameFilter()
{
  this->AddToList("One of FeedbackContourTool's feedback nodes");
  this->AddToList(ContourTool::MIDAS_CONTOUR_TOOL_BACKGROUND_CONTOUR);
  this->AddToList(Tool::SEEDS_NAME);
  this->AddToList(Tool::CONTOURS_NAME);
  this->AddToList(Tool::REGION_GROWING_NAME);
  this->AddToList(Tool::PRIOR_CONTOURS_NAME);
  this->AddToList(Tool::NEXT_CONTOURS_NAME);
  this->AddToList(Tool::DRAW_CONTOURS_NAME);
  this->AddToList(PaintbrushTool::EROSIONS_SUBTRACTIONS_NAME);
  this->AddToList(PaintbrushTool::EROSIONS_ADDITIONS_NAME);
  this->AddToList(PaintbrushTool::DILATIONS_SUBTRACTIONS_NAME);
  this->AddToList(PaintbrushTool::DILATIONS_ADDITIONS_NAME);
  this->AddToList(MorphologicalSegmentorPipelineManager::SEGMENTATION_OF_LAST_STAGE_NAME);
  this->AddToList(PolyTool::MIDAS_POLY_TOOL_ANCHOR_POINTS);
  this->AddToList(PolyTool::MIDAS_POLY_TOOL_PREVIOUS_CONTOUR);
  this->AddToList("Paintbrush_Node");
  this->AddToList("widget1Plane");
  this->AddToList("widget2Plane");
  this->AddToList("widget3Plane");
  this->SetPropertyName("name");
}


//-----------------------------------------------------------------------------
ToolWorkingDataNameFilter::~ToolWorkingDataNameFilter()
{

}

}
