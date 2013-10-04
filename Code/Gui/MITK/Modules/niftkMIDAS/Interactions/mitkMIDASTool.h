/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkMIDASTool_h
#define mitkMIDASTool_h

#include "niftkMIDASExports.h"
#include "mitkMIDASPointSetInteractor.h"
#include <mitkFeedbackContourTool.h>
#include <mitkPointSet.h>
#include <mitkDataNode.h>
#include <mitkPositionEvent.h>
#include <mitkMessage.h>

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
  class NIFTKMIDAS_EXPORT MIDASTool : public FeedbackContourTool {

  public:

    mitkClassMacro(MIDASTool, FeedbackContourTool);
    const char* GetGroup() const;

    /// \brief We store a seed point set name, so all classes have access to the name.
    static const std::string SEED_POINT_SET_NAME;

    /// \brief We store the name of the current slice contours, so all classes have access to the name.
    static const std::string CURRENT_CONTOURS_NAME;

    /// \brief We store the name of the region growing image, so all classes have access to the name.
    static const std::string REGION_GROWING_IMAGE_NAME;

    /// \brief We store the name of the prior contours, so all classes have access to the name.
    static const std::string PRIOR_CONTOURS_NAME;

    /// \brief We store the name of the next contours, so all classes have access to the name.
    static const std::string NEXT_CONTOURS_NAME;

    /// \brief We store the name of the draw tool contours, so all classes have access to the name.
    static const std::string DRAW_CONTOURS_NAME;

    /// \brief We store the name of the initial segmentation image, so all classes have access to the name.
    static const std::string INITIAL_SEGMENTATION_IMAGE_NAME;

    /// \brief We store the name of the initial set of seeds, so all classes have access to the name.
    static const std::string INITIAL_SEEDS_NAME;

    /// \brief We store the name of the MIDAS connection breaker image, used in Morphological Editor.
    static const std::string MORPH_EDITS_EROSIONS_SUBTRACTIONS;

    /// \brief We store the name of the MIDAS additions image, used in Morphological Editor.
    static const std::string MORPH_EDITS_EROSIONS_ADDITIONS;

    /// \brief We store the name of the MIDAS connection breaker image, used in Morphological Editor.
    static const std::string MORPH_EDITS_DILATIONS_SUBTRACTIONS;

    /// \brief We store the name of the MIDAS additions image, used in Morphological Editor.
    static const std::string MORPH_EDITS_DILATIONS_ADDITIONS;

    // Currently, creating state machine using hard coded string, as I don't know where to load them from.
    static const std::string MIDAS_SEED_TOOL_STATE_MACHINE_XML;

    // Currently, creating state machine using hard coded string, as I don't know where to load them from.
    static const std::string MIDAS_SEED_DROPPER_STATE_MACHINE_XML;

    // Currently, creating state machine using hard coded string, as I don't know where to load them from.
    static const std::string MIDAS_POLY_TOOL_STATE_MACHINE_XML;

    // Currently, creating state machine using hard coded string, as I don't know where to load them from.
    static const std::string MIDAS_DRAW_TOOL_STATE_MACHINE_XML;

    // Currently, creating state machine using hard coded string, as I don't know where to load them from.
    static const std::string MIDAS_PAINTBRUSH_TOOL_STATE_MACHINE_XML;

    // Currently, creating state machine using hard coded string, as I don't know where to load them from.
    static const std::string MIDAS_TOOL_KEYPRESS_STATE_MACHINE_XML;

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

    /// \see mitk::StateMachine::CanHandleEvent
    float CanHandleEvent(const StateEvent *) const;

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

    // Stores the current display interactor configurations when this tool is activated.
    // The configurations are restored when the tool is deactivated.
    std::map<mitk::ServiceReference, mitk::EventConfig> m_DisplayInteractorConfigs;

  };//class

}//namespace

#endif
