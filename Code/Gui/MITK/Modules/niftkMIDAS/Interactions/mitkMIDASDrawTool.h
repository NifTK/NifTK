/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkMIDASDrawTool_h
#define mitkMIDASDrawTool_h

#include "niftkMIDASExports.h"
#include "mitkMIDASContourTool.h"
#include "mitkMIDASDrawToolEventInterface.h"

namespace mitk {

/**
 * \class MIDASDrawTool
 * \brief Tool to draw lines around voxel edges like MIDAS does rather than through them
 * as most of the MITK tools do.
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
class NIFTKMIDAS_EXPORT MIDASDrawTool : public MIDASContourTool {

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

  /// \brief Different to MIDASContourTool::ClearData which clears the Feedback contour, this one finds the working data node, and erases all contours.
  virtual void ClearWorkingData();

  /// \brief Called by the main application to clean the contour, which means, to erase any bits of contour
  /// not currently touching the region growing image.
  virtual void Clean(const int& sliceNumber, const int& axisNumber);

protected:

  MIDASDrawTool(); // purposely hidden
  virtual ~MIDASDrawTool(); // purposely hidden

  /// \see mitk::MIDASStateMachine::CanHandle
  virtual float CanHandle(const mitk::StateEvent* stateEvent) const;

  /**
  \brief Called when the tool gets activated (registered to mitk::GlobalInteraction).

  Derived tools should call their parents implementation.
  */
  virtual void Activated();

private:

  template<typename TPixel, unsigned int VImageDimension>
  void ITKCleanContours(
      itk::Image<TPixel, VImageDimension> *itkImage,
      mitk::ContourModelSet& inputContours,
      mitk::ContourModelSet& outputContours,
      const int& axis,
      const int& sliceNumber
      );

  /// \brief Operation constant, used in Undo/Redo framework.
  static const mitk::OperationType MIDAS_DRAW_TOOL_OP_ERASE_CONTOUR;

  /// \brief Operation constant, used in Undo/Redo framework.
  static const mitk::OperationType MIDAS_DRAW_TOOL_OP_CLEAN_CONTOUR;

  /// \brief Internal method to delete from the mitkToolManager WorkingData[workingDataNumber], which should be a mitk::ContourModelSet representing the "currentContours" ie Green lines in MIDAS.
  bool DeleteFromContour(const int &workingDataNumber, Action* action, const StateEvent* stateEvent);

  /// \brief Cursor size for editing, currently called "Eraser" in MIDAS, where this eraser is defined in millimetres distance.
  int m_CursorSize;

  /// \brief Stores the most recent point, (i.e. the end of the line if we are drawing a line).
  mitk::Point3D m_MostRecentPointInMillimetres;

  /// \brief Pointer to interface object, used as callback in Undo/Redo framework
  MIDASDrawToolEventInterface::Pointer m_Interface;

};

}

#endif
