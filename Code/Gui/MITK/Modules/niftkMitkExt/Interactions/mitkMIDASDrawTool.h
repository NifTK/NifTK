/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-19 22:31:43 +0000 (Sat, 19 Nov 2011) $
 Revision          : $Revision: 7815 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef MITKMIDASDRAWTOOL_H
#define MITKMIDASDRAWTOOL_H

#include "niftkMitkExtExports.h"
#include "mitkMIDASContourTool.h"
#include "mitkMIDASDrawToolEventInterface.h"

namespace mitk {

  /**
   * \class MIDASDrawTool
   * \brief Tool to draw lines around voxel edges like MIDAS does rather than through them as most of the MITK tools do.
   *
   * Provides
   * <pre>
   * 1. Left mouse button down = start line
   * 2. Move with left button down = continue line
   * 3. Left mouse button release = finish line
   * 4. Middle mouse button down = erase anything within the range given by the Eraser (cursor size).
   * 5. Move with middle mouse button down = same as 4.
   * 6. Middle mouse button release = finish editing.
   * </pre>
   * and includes Undo/Redo functionality.
   */
  class NIFTKMITKEXT_EXPORT MIDASDrawTool : public MIDASContourTool {

  public:

    mitkClassMacro(MIDASDrawTool, MIDASContourTool);
    itkNewMacro(MIDASDrawTool);

    /// \brief Method to enable this class to interact with the Undo/Redo framework.
    virtual void ExecuteOperation(Operation* operation);

    /// \see mitk::Tool::GetName()
    virtual const char* GetName() const;

    /// \see mitk::Tool::GetXPM()
    virtual const char** GetXPM() const;

    /// \brief Get the Cursor size, default 1.
    itkGetConstMacro(CursorSize, int);

    /// \brief Set the cursor size, default 1.
    void SetCursorSize(int current);

    /// \brief Used to send messages when the cursor size is changed or should be updated in a GUI. */
    Message1<int> CursorSizeChanged;

    /// \brief Start drawing a line at the given mouse point.
    virtual bool OnLeftMousePressed  (Action* action, const StateEvent* stateEvent);

    /// \brief Continue drawing a line.
    virtual bool OnLeftMouseMoved    (Action* action, const StateEvent* stateEvent);

    /// \brief Finish drawing a line.
    virtual bool OnLeftMouseReleased (Action* action, const StateEvent* stateEvent);

    /// \brief Erase any contours within the distance given by the cursor size in this class, and denoted by the Erase slider in the GUI.
    virtual bool OnMiddleMousePressed(Action* action, const StateEvent* stateEvent);

    /// \brief Erase any contours within the distance given by the cursor size in this class, and denoted by the Erase slider in the GUI.
    virtual bool OnMiddleMouseMoved  (Action* action, const StateEvent* stateEvent);

    /// \brief Finish editing.
    virtual bool OnMiddleMouseReleased (Action* action, const StateEvent* stateEvent);

  protected:

    MIDASDrawTool(); // purposely hidden
    virtual ~MIDASDrawTool(); // purposely hidden

  private:

    /// \brief Internal method to delete from the mitkToolManager WorkingData, data set 2, which should be a mitk::ContourSet representing the "currentContours" ie Green lines in MIDAS.
    bool DeleteFromContour(Action* action, const StateEvent* stateEvent);

    /// \brief Cursor size for editing, currently called "Eraser" in MIDAS, where this eraser is defined in millimetres distance.
    int m_CursorSize;

    /// \brief Stores the most recent point, (i.e. the end of the line if we are drawing a line).
    mitk::Point3D m_MostRecentPointInMillimetres;

    /// \brief Operation constant, used in Undo/Redo framework.
    static const mitk::OperationType MIDAS_DRAW_TOOL_OP_ERASE_CONTOUR;

    /// \brief Pointer to interface object, used as callback in Undo/Redo framework
    MIDASDrawToolEventInterface::Pointer m_Interface;

  };//class


}//namespace

#endif
