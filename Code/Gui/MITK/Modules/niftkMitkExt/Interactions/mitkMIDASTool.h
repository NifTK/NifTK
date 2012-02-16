/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-21 08:53:21 +0100 (Wed, 21 Sep 2011) $
 Revision          : $Revision: 7344 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef MITKMIDASTOOL_H
#define MITKMIDASTOOL_H

#include "niftkMitkExtExports.h"
#include "mitkFeedbackContourTool.h"
#include "mitkPointSet.h"
#include "mitkDataNode.h"
#include "mitkMIDASPointSetInteractor.h"
#include "mitkPositionEvent.h"

namespace mitk {

  /**
   * \class MIDASTool
   * \brief Base class for MIDAS tools that need access to the list of
   * seeds for the current reference data volume registered with the ToolManager.
   * I made it inherit from FeedbackContourTool, as multiple inheritance was getting messy.
   */
  class NIFTKMITKEXT_EXPORT MIDASTool : public FeedbackContourTool {

  public:

    mitkClassMacro(MIDASTool, FeedbackContourTool);
    const char* GetGroup() const;

    // We store a seed point set name, so all classes have access to it.
    static const std::string SEED_POINT_SET_NAME;

    // We store a name for region growing image used in the general editor.
    static const std::string REGION_GROWING_IMAGE_NAME;

    // We store a name for see prior image used in the general editor.
    static const std::string SEE_PRIOR_IMAGE_NAME;

    // We store a name for see next image used in the general editor.
    static const std::string SEE_NEXT_IMAGE_NAME;

    // When called, we get a reference to the set of seeds, and set up the interactor(s).
    virtual void Activated();

    // When called, we unregister the reference to the set of seeds, and deactivate the interactors(s).
    virtual void Deactivated();

    // Wipe's any tool specific data, such as contours, seed points etc.
    virtual void Wipe();

  protected:

    MIDASTool(); // purposefully hidden
    MIDASTool(const char* type); // purposefully hidden
    virtual ~MIDASTool(); // purposely hidden

    // Makes the current window re-render
    virtual void RenderCurrentWindow(const PositionEvent& event);

    // Makes all windows re-render
    virtual void RenderAllWindows();

    // Can be called by derived classes to try and set the point set
    virtual void FindPointSet(mitk::PointSet*& pointSet, mitk::DataNode*& pointSetNode);

    // Helper method to update a boolean property on a given working image.
    virtual void UpdateWorkingImageBooleanProperty(int workingImageNumber, std::string name, bool value);

  private:

    // This is the interactor just to add points. All MIDAS tools can add seeds. Only the SeedTool can move/remove them.
    mitk::MIDASPointSetInteractor::Pointer m_AddToPointSetInteractor;

  };//class

}//namespace

#endif
