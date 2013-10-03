/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkMIDASDataNodeNameStringFilter_h
#define mitkMIDASDataNodeNameStringFilter_h

#include "niftkDnDDisplayExports.h"
#include <mitkDataNodeStringPropertyFilter.h>
#include <mitkDataStorage.h>

namespace mitk
{

/**
 * \class MIDASDataNodeNameStringFilter
 *
 * \brief A filter that returns Pass=false if the name is in the following list, and true otherwise.
 *
 * </pre>
 * FeedbackContourTool
 * MIDASContourTool
 * mitk::MIDASTool::SEED_POINT_SET_NAME
 * mitk::MIDASTool::CURRENT_CONTOURS_NAME
 * mitk::MIDASTool::REGION_GROWING_IMAGE_NAME
 * mitk::MIDASTool::PRIOR_CONTOURS_NAME
 * mitk::MIDASTool::NEXT_CONTOURS_NAME
 * mitk::MIDASTool::MORPH_EDITS_SUBTRACTIONS
 * mitk::MIDASTool::MORPH_EDITS_ADDITIONS
 * mitk::MIDASPolyTool::MIDAS_POLY_TOOL_ANCHOR_POINTS
 * mitk::MIDASPolyTool::MIDAS_POLY_TOOL_PREVIOUS_CONTOUR
 * Paintbrush_Node
 * </pre>
 */
class NIFTKDNDDISPLAY_EXPORT MIDASDataNodeNameStringFilter : public mitk::DataNodeStringPropertyFilter
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

}; // end class

} // end namespace

#endif


