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
#include "mitkMessage.h"

namespace mitk {

  /**
   * \class MIDASTool
   * \brief Base class for MIDAS tools that need access to the list of
   * seeds for the current reference data volume registered with the ToolManager.
   *
   * Matt: I made it inherit from FeedbackContourTool, as multiple inheritance was getting messy.
   *
   * Note that the MIDASSeedTool, MIDASDrawTool and MIDASPolyTool all inherit from this class.
   * Each of these tools will have their MIDASPointSetInteractor. Each tool is managed
   * by an mitk::ToolManager which guarantees that only one is active at any given time.
   * As each tool becomes activated it will register the interactor with GlobalInteraction,
   * and as the tool becomes deactivated it will de-register the interactor with GlobalInteraction.
   *
   * In addition (as of 18th May 2012), this class keeps track of the number of seeds
   * and calls OnNumberOfSeedsChanged(int numberOfSeeds) when the number of seeds changed.
   * This means derived classes could be notified when the number of seeds has changed.
   * This is only called if the tool is Active.
   *
   * \sa mitk::MIDASSeedTool
   * \sa mitk::MIDASContourTool
   * \sa mitk::MIDASDrawTool
   * \sa mitk::MIDASPolyTool
   * \sa mitk::MIDASPointSetInteractor
   */
  class NIFTKMITKEXT_EXPORT MIDASTool : public FeedbackContourTool {

  public:

    mitkClassMacro(MIDASTool, FeedbackContourTool);
    const char* GetGroup() const;

    /// \brief We store a seed point set name, so all classes have access to the name.
    static const std::string SEED_POINT_SET_NAME;

    /// \brief We store the name of the current slice contours, so all class have access to the name.
    static const std::string CURRENT_CONTOURS_NAME;

    /// \brief We store the name of the region growing image, so all class have access to the name.
    static const std::string REGION_GROWING_IMAGE_NAME;

    /// \brief We store the name of the prior contours, so all class have access to the name.
    static const std::string PRIOR_CONTOURS_NAME;

    /// \brief We store the name of the next contours, so all class have access to the name.
    static const std::string NEXT_CONTOURS_NAME;

    /// \brief When called, we get a reference to the set of seeds, and set up the interactor(s).
    virtual void Activated();

    /// \brief When called, we unregister the reference to the set of seeds, and deactivate the interactors(s).
    virtual void Deactivated();

    /// \brief Used to signal that the number of seeds has changed
    Message1<int> NumberOfSeedsHasChanged;

    /// \brief Set the flag to block the signal that indicates that the number of seeds has changed.
    itkSetMacro(BlockNumberOfSeedsSignal, bool);

    /// \brief Get the flag to block the signal that indicates that the number of seeds has changed.
    itkGetMacro(BlockNumberOfSeedsSignal, bool);

  protected:

    MIDASTool(); // purposefully hidden
    MIDASTool(const char* type); // purposefully hidden
    virtual ~MIDASTool(); // purposely hidden

    /// \brief Makes the current window re-render
    virtual void RenderCurrentWindow(const PositionEvent& event);

    /// \brief Makes all windows re-render
    virtual void RenderAllWindows();

    /// \brief Can be called by derived classes to try and set the point set
    virtual void FindPointSet(mitk::PointSet*& pointSet, mitk::DataNode*& pointSetNode);

    /// \brief Helper method to update a boolean property on a given working data node.
    virtual void UpdateWorkingDataNodeBooleanProperty(int workingDataNodeNumber, std::string name, bool value);

  private:

    /// \brief Called when the seeds have been modified.
    void OnSeedsModified();

    // This is the interactor just to add points. All MIDAS tools can add seeds. Only the SeedTool can move/remove them.
    mitk::MIDASPointSetInteractor::Pointer m_AddToPointSetInteractor;

    // Used to track when the number of seeds changes.
    int m_LastSeenNumberOfSeeds;

    // Used to track when the number of seeds changes.
    unsigned long m_SeedsChangedTag;

    //  Track whether this tool is activated or not.
    bool m_IsActivated;

    // To control if we block the NumberOfSeedsHasChanged signal.
    bool m_BlockNumberOfSeedsSignal;

  };//class

}//namespace

#endif
