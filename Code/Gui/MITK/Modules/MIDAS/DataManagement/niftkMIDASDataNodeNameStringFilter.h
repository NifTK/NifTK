/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __niftkMIDASDataNodeNameStringFilter_h
#define __niftkMIDASDataNodeNameStringFilter_h

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
 * MIDASContourTool
 * niftk::MIDASTool::SEED_POINT_SET_NAME
 * niftk::MIDASTool::CURRENT_CONTOURS_NAME
 * niftk::MIDASTool::REGION_GROWING_IMAGE_NAME
 * niftk::MIDASTool::PRIOR_CONTOURS_NAME
 * niftk::MIDASTool::NEXT_CONTOURS_NAME
 * niftk::MIDASTool::MORPH_EDITS_SUBTRACTIONS
 * niftk::MIDASTool::MORPH_EDITS_ADDITIONS
 * niftk::MIDASPolyTool::MIDAS_POLY_TOOL_ANCHOR_POINTS
 * niftk::MIDASPolyTool::MIDAS_POLY_TOOL_PREVIOUS_CONTOUR
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


