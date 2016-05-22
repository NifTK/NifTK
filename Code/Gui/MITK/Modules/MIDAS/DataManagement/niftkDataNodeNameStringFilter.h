/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkDataNodeNameStringFilter_h
#define niftkDataNodeNameStringFilter_h

#include "niftkMIDASExports.h"

#include <mitkDataNodeStringPropertyFilter.h>
#include <mitkDataStorage.h>

namespace niftk
{

/**
 * \class MIDASDataNodeNameStringFilter
 *
 * \brief A filter that returns Pass=false if the name is in the following list, and true otherwise.
 *
 * </pre>
 * FeedbackContourTool
 * niftkContourTool
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
class NIFTKMIDAS_EXPORT MIDASDataNodeNameStringFilter : public mitk::DataNodeStringPropertyFilter
{

public:

  mitkClassMacro(MIDASDataNodeNameStringFilter, mitk::DataNodeStringPropertyFilter);
  itkNewMacro(MIDASDataNodeNameStringFilter);

protected:

  MIDASDataNodeNameStringFilter();
  virtual ~MIDASDataNodeNameStringFilter();

  MIDASDataNodeNameStringFilter(const MIDASDataNodeNameStringFilter&); // Purposefully not implemented.
  MIDASDataNodeNameStringFilter& operator=(const MIDASDataNodeNameStringFilter&); // Purposefully not implemented.

private:

};

}

#endif


