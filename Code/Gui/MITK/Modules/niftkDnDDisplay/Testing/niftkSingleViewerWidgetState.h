/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __niftkSingleViewerWidgetState_h
#define __niftkSingleViewerWidgetState_h

#include <niftkSingleViewerWidget.h>


class niftkSingleViewerWidgetState : public itk::Object
{
public:

  mitkClassMacro(niftkSingleViewerWidgetState, itk::Object);
  mitkNewMacro1Param(niftkSingleViewerWidgetState, const niftkSingleViewerWidget*);
  mitkNewMacro1Param(niftkSingleViewerWidgetState, Self::Pointer);

  /// \brief Gets the orientation of the viewer.
  itkGetConstMacro(Orientation, MIDASOrientation);

  /// \brief Sets the orientation of the viewer.
  itkSetMacro(Orientation, MIDASOrientation);

  /// \brief Gets the window layout of the viewer.
  itkGetConstMacro(WindowLayout, WindowLayout);

  /// \brief Sets the window of the viewer.
  itkSetMacro(WindowLayout, WindowLayout);

  /// \brief Gets the selected render window of the viewer.
  itkGetConstMacro(SelectedRenderWindow, QmitkRenderWindow*);

  /// \brief Sets the selected render window of the viewer.
  itkSetObjectMacro(SelectedRenderWindow, QmitkRenderWindow);

  /// \brief Gets the selected time step in the viewer.
  itkGetConstMacro(TimeStep, unsigned);

  /// \brief Sets the selected time step in the viewer.
  itkSetMacro(TimeStep, unsigned);

  /// \brief Gets the selected position in the viewer.
  itkGetConstMacro(SelectedPosition, mitk::Point3D);

  /// \brief Sets the selected position in the viewer.
  itkSetMacro(SelectedPosition, mitk::Point3D);

  /// \brief Gets the cursor positions in the render windows of the viewer.
  itkGetConstMacro(CursorPositions, std::vector<mitk::Vector2D>);

  /// \brief Sets the cursor positions in the render windows of the viewer.
  void SetCursorPositions(const std::vector<mitk::Vector2D>& cursorPositions)
  {
    this->m_CursorPositions = cursorPositions;
  }

  /// \brief Gets the scale factors in the render windows of the viewer.
  itkGetConstMacro(ScaleFactors, std::vector<double>);

  /// \brief Sets the scale factors in the render windows of the viewer.
  void SetScaleFactors(const std::vector<double>& scaleFactors)
  {
    this->m_ScaleFactors = scaleFactors;
  }

  /// \brief Gets the cursor position binding property of the viewer.
  itkGetConstMacro(CursorPositionBinding, bool);

  /// \brief Sets the cursor position binding property of the viewer.
  itkSetMacro(CursorPositionBinding, bool);

  /// \brief Gets the scale factor binding property of the viewer.
  itkGetConstMacro(ScaleFactorBinding, bool);

  /// \brief Sets the scale factor binding property of the viewer.
  itkSetMacro(ScaleFactorBinding, bool);

  bool operator==(const niftkSingleViewerWidgetState& otherState) const
  {
    return
        this->GetOrientation() == otherState.GetOrientation()
        && this->GetWindowLayout() == otherState.GetWindowLayout()
        && this->GetSelectedRenderWindow() == otherState.GetSelectedRenderWindow()
        && this->GetTimeStep() == otherState.GetTimeStep()
        && this->GetSelectedPosition() == otherState.GetSelectedPosition()
        && this->GetCursorPositions() == otherState.GetCursorPositions()
        && this->GetScaleFactors() == otherState.GetScaleFactors()
        && this->GetCursorPositionBinding() == otherState.GetCursorPositionBinding()
        && this->GetScaleFactorBinding() == otherState.GetScaleFactorBinding();
  }

  inline bool operator!=(const niftkSingleViewerWidgetState& otherState) const
  {
    return !(*this == otherState);
  }

  void PrintDifference(niftkSingleViewerWidgetState::Pointer otherState, std::ostream & os = std::cout, itk::Indent indent = 0) const
  {
    if (this->GetOrientation() != otherState->GetOrientation())
    {
      os << indent << "Orientation: " << this->GetOrientation() << " ; " << otherState->GetOrientation() << std::endl;
    }
    if (this->GetWindowLayout() != otherState->GetWindowLayout())
    {
      os << indent << "Window layout: " << this->GetWindowLayout() << " ; " << otherState->GetWindowLayout() << std::endl;
    }
    if (this->GetSelectedRenderWindow() != otherState->GetSelectedRenderWindow())
    {
      os << indent << "Selected render window: " << this->GetSelectedRenderWindow() << " ; " << otherState->GetOrientation() << std::endl;
    }
    if (this->GetTimeStep() != otherState->GetTimeStep())
    {
      os << indent << "Time step: " << this->GetTimeStep() << " ; " << otherState->GetTimeStep() << std::endl;
    }
    if (this->GetSelectedPosition() != otherState->GetSelectedPosition())
    {
      os << indent << "Selected position: " << this->GetSelectedPosition() << " ; " << otherState->GetSelectedPosition() << std::endl;
    }
    if (this->GetCursorPositions() != otherState->GetCursorPositions())
    {
      os << indent << "Cursor positions:" << std::endl;
      std::vector<mitk::Vector2D>::const_iterator cursorPositionsIt = m_CursorPositions.begin();
      std::vector<mitk::Vector2D>::const_iterator cursorPositionsEnd = m_CursorPositions.end();
      for ( ; cursorPositionsIt != cursorPositionsEnd; ++cursorPositionsIt)
      {
        os << indent << "    " << *cursorPositionsIt << std::endl;
      }
      cursorPositionsIt = otherState->GetCursorPositions().begin();
      cursorPositionsEnd = otherState->GetCursorPositions().end();
      for ( ; cursorPositionsIt != cursorPositionsEnd; ++cursorPositionsIt)
      {
        os << indent << "    " << *cursorPositionsIt << std::endl;
      }
    }
    if (this->GetScaleFactors() != otherState->GetScaleFactors())
    {
      os << indent << "Scale factors:" << std::endl;
      std::vector<double>::const_iterator scaleFactorsIt = m_ScaleFactors.begin();
      std::vector<double>::const_iterator scaleFactorsEnd = m_ScaleFactors.end();
      for ( ; scaleFactorsIt != scaleFactorsEnd; ++scaleFactorsIt)
      {
        os << indent << "    " << *scaleFactorsIt << std::endl;
      }
      scaleFactorsIt = otherState->GetScaleFactors().begin();
      scaleFactorsEnd = otherState->GetScaleFactors().end();
      for ( ; scaleFactorsIt != scaleFactorsEnd; ++scaleFactorsIt)
      {
        os << indent << "    " << *scaleFactorsIt << std::endl;
      }
    }
    if (this->GetCursorPositionBinding() != otherState->GetCursorPositionBinding())
    {
      os << indent << "Cursor position binding: " << this->GetCursorPositionBinding() << " ; " << otherState->GetCursorPositionBinding() << std::endl;
    }
    if (this->GetScaleFactorBinding() != otherState->GetScaleFactorBinding())
    {
      os << indent << "Scale factor binding: " << this->GetScaleFactorBinding() << " ; " << otherState->GetScaleFactorBinding() << std::endl;
    }
  }

protected:

  /// \brief Constructs a niftkSingleViewerWidgetState object that stores the current state of the specified viewer.
  niftkSingleViewerWidgetState(const niftkSingleViewerWidget* viewer)
  : itk::Object()
  , m_Orientation(viewer->GetOrientation())
  , m_WindowLayout(viewer->GetWindowLayout())
  , m_SelectedRenderWindow(viewer->GetSelectedRenderWindow())
  , m_TimeStep(viewer->GetTimeStep())
  , m_SelectedPosition(viewer->GetSelectedPosition())
  , m_CursorPositions(viewer->GetCursorPositions())
  , m_ScaleFactors(viewer->GetScaleFactors())
  , m_CursorPositionBinding(viewer->GetCursorPositionBinding())
  , m_ScaleFactorBinding(viewer->GetScaleFactorBinding())
  {
  }

  /// \brief Constructs a niftkSingleViewerWidgetState object as a copy of another state object.
  niftkSingleViewerWidgetState(Self::Pointer otherState)
  : itk::Object()
  , m_Orientation(otherState->GetOrientation())
  , m_WindowLayout(otherState->GetWindowLayout())
  , m_SelectedRenderWindow(otherState->GetSelectedRenderWindow())
  , m_TimeStep(otherState->GetTimeStep())
  , m_SelectedPosition(otherState->GetSelectedPosition())
  , m_CursorPositions(otherState->GetCursorPositions())
  , m_ScaleFactors(otherState->GetScaleFactors())
  , m_CursorPositionBinding(otherState->GetCursorPositionBinding())
  , m_ScaleFactorBinding(otherState->GetScaleFactorBinding())
  {
  }

  /// \brief Destructs a niftkSingleViewerWidgetState object.
  virtual ~niftkSingleViewerWidgetState()
  {
  }

  /// \brief Prints the collected signals to the given stream or to the standard output if no stream is given.
  virtual void PrintSelf(std::ostream & os, itk::Indent indent) const
  {
    os << indent << "orientation: " << m_Orientation << std::endl;
    os << indent << "window layout: " << m_WindowLayout << std::endl;
    os << indent << "selected render window: " << m_SelectedRenderWindow << ", " << m_SelectedRenderWindow->GetRenderer()->GetName() << std::endl;
    os << indent << "time step: " << m_TimeStep << std::endl;
    os << indent << "selected position: " << m_SelectedPosition << std::endl;
    os << indent << "cursor positions: " << std::endl;
    std::vector<mitk::Vector2D>::const_iterator cursorPositionsIt = m_CursorPositions.begin();
    std::vector<mitk::Vector2D>::const_iterator cursorPositionsEnd = m_CursorPositions.end();
    for ( ; cursorPositionsIt != cursorPositionsEnd; ++cursorPositionsIt)
    {
      os << indent << "    " << *cursorPositionsIt << std::endl;
    }
    os << indent << "scale factors: " << std::endl;
    std::vector<mitk::Vector2D>::const_iterator scaleFactorsIt = m_CursorPositions.begin();
    std::vector<mitk::Vector2D>::const_iterator scaleFactorsEnd = m_CursorPositions.end();
    for ( ; scaleFactorsIt != scaleFactorsEnd; ++scaleFactorsIt)
    {
      os << indent << "    " << *scaleFactorsIt << std::endl;
    }
    os << indent << "cursor position binding: " << m_CursorPositionBinding << std::endl;
    os << indent << "scale factor binding: " << m_ScaleFactorBinding << std::endl;
  }

private:

  /// \brief The orientation of the viewer.
  MIDASOrientation m_Orientation;

  /// \brief The window layout of the viewer.
  WindowLayout m_WindowLayout;

  /// \brief The selected render window of the viewer.
  QmitkRenderWindow* m_SelectedRenderWindow;

  /// \brief The selected time step in the viewer.
  unsigned m_TimeStep;

  /// \brief The selected position in the viewer.
  mitk::Point3D m_SelectedPosition;

  /// \brief The cursor positions in the render windows of the viewer.
  std::vector<mitk::Vector2D> m_CursorPositions;

  /// \brief The scale factors in the render windows of the viewer.
  std::vector<double> m_ScaleFactors;

  /// \brief The cursor binding property of the viewer.
  bool m_CursorPositionBinding;

  /// \brief The scale factor binding property of the viewer.
  bool m_ScaleFactorBinding;

};

#endif
