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

#ifndef MITKMIDASDATANODENAMESTRINGPROPERTYFILTER_H
#define MITKMIDASDATANODENAMESTRINGPROPERTYFILTER_H

#include "niftkMitkExtExports.h"

#include "mitkDataNodeStringPropertyFilter.h"
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
class NIFTKMITKEXT_EXPORT MIDASDataNodeNameStringFilter : public mitk::DataNodeStringPropertyFilter
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


