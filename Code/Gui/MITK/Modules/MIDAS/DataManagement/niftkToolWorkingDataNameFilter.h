/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkToolWorkingDataNameFilter_h
#define niftkToolWorkingDataNameFilter_h

#include "niftkMIDASExports.h"

#include <mitkDataNodeStringPropertyFilter.h>
#include <mitkDataStorage.h>

namespace niftk
{

/**
 * \class ToolWorkingDataNameFilter
 *
 * \brief A filter that returns Pass=false if the name is in the following list, and true otherwise.
 *
 * </pre>
 * FeedbackContourTool
 * niftk::ContourTool::MIDAS_CONTOUR_TOOL_BACKGROUND_CONTOUR
 * niftk::Tool::SEED_POINT_SET_NAME
 * niftk::Tool::CURRENT_CONTOURS_NAME
 * niftk::Tool::REGION_GROWING_IMAGE_NAME
 * niftk::Tool::PRIOR_CONTOURS_NAME
 * niftk::Tool::NEXT_CONTOURS_NAME
 * niftk::Tool::MORPH_EDITS_SUBTRACTIONS
 * niftk::Tool::MORPH_EDITS_ADDITIONS
 * niftk::PolyTool::MIDAS_POLY_TOOL_ANCHOR_POINTS
 * niftk::PolyTool::MIDAS_POLY_TOOL_PREVIOUS_CONTOUR
 * Paintbrush_Node
 * </pre>
 */
class NIFTKMIDAS_EXPORT ToolWorkingDataNameFilter : public mitk::DataNodeStringPropertyFilter
{

public:

  mitkClassMacro(ToolWorkingDataNameFilter, mitk::DataNodeStringPropertyFilter);
  itkNewMacro(ToolWorkingDataNameFilter);

protected:

  ToolWorkingDataNameFilter();
  virtual ~ToolWorkingDataNameFilter();

  ToolWorkingDataNameFilter(const ToolWorkingDataNameFilter&); // Purposefully not implemented.
  ToolWorkingDataNameFilter& operator=(const ToolWorkingDataNameFilter&); // Purposefully not implemented.

private:

};

}

#endif


