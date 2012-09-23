/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-08-02 05:55:41 +0100 (Tue, 02 Aug 2011) $
 Revision          : $Revision: 6917 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef MITKMIDASPAINTBRUSHTOOL_H
#define MITKMIDASPAINTBRUSHTOOL_H

#include "niftkMitkExtExports.h"
#include "mitkOperation.h"
#include "mitkOperationActor.h"
#include "mitkSegTool2D.h"
#include "mitkImage.h"
#include "mitkGeometry3D.h"
#include "mitkMIDASPaintbrushToolOpEditImage.h"
#include "mitkMIDASPaintbrushToolEventInterface.h"
#include "itkImage.h"
#include "itkImageUpdatePixelWiseSingleValueProcessor.h"

namespace mitk
{

 /**
  * \class MIDASPaintbrushTool
  * \brief MIDAS paint brush tool used during editing on the morphological editor screen (a.k.a connection breaker).
  *
  * Note the following:
  * <pre>
  * 1.) Writes into 4 images, so ToolManager must have 4 working volume to edit into.
  *     We define Working Image[0] = "additions image for erosions", which is added to the main segmentation to add stuff back into the volume.
  *     We define Working Image[1] = "subtractions image for erosions", which is subtracted from the main segmentation to do connection breaking.
  *     We define Working Image[2] = "additions image for dilations", which is added to the main segmentation to add stuff back into the volume.
  *     We define Working Image[3] = "subtractions image for dilations", which is subtracted from the main segmentation to do connection breaking.
  * 2.) Then:
  *     Left mouse = paint into the "additions image".
  *     Middle mouse = paint into the "subtractions image".
  *     Right mouse = subtract from the "subtractions image".
  * 3.) We derive from SegTool2D to keep things simple, as we just need to convert from mm world points to voxel points, and paint.
  * 4.) Derives from mitk::OperationActor, so this tool supports undo/redo.
  * </pre>
  *
  * This class is a MITK tool with a GUI defined in QmitkMIDASPaintbrushToolGUI, and instantiated
  * using the object factory described in Maleike et. al. doi:10.1016/j.cmpb.2009.04.004.
  *
  * To effectively use this tool, you need a 3 button mouse.
  *
  * Trac 1695, 1700, 1701, 1706: Fixing up dilations: We change pipeline so that WorkingData 0,1 are
  * applied during erosions phase, and WorkingData 2,3 are applied during dilations phase.
  */
class NIFTKMITKEXT_EXPORT MIDASPaintbrushTool : public SegTool2D
{

public:
  mitkClassMacro(MIDASPaintbrushTool, SegTool2D);
  itkNewMacro(MIDASPaintbrushTool);

  typedef itk::Image<mitk::Tool::DefaultSegmentationDataType, 3> ImageType;
  typedef itk::ImageUpdatePixelWiseSingleValueProcessor<mitk::Tool::DefaultSegmentationDataType, 3> ProcessorType;

  /** Strings to help the tool identify itself in GUI. */
  virtual const char* GetName() const;
  virtual const char** GetXPM() const;

  /** We store the name of a property that stores the image region. */
  static const std::string REGION_PROPERTY_NAME;

  /** Get the Cursor size, default 1. */
  itkGetConstMacro(CursorSize, int);

  /** Set the cursor size, default 1. */
  void SetCursorSize(int current);

  /** If true, we are editing image 0,1, and if false, we are editing image 2,3. Default true. */
  itkSetMacro(ErosionMode, bool);

  /** If true, we are editing image 0,1, and if false, we are editing image 2,3. Default true. */
  itkGetMacro(ErosionMode, bool);

  /** Used to send messages when the cursor size is changed or should be updated in a GUI. */
  Message1<int> CursorSizeChanged;

  /** Method to enable this class to interact with the Undo/Redo framework. */
  virtual void ExecuteOperation(Operation* operation);

  /** \see mitk::StateMachine::CanHandleEvent */
  float CanHandleEvent(const StateEvent *) const;

  /** Process all mouse events. */
  virtual bool OnLeftMousePressed   (Action* action, const StateEvent* stateEvent);
  virtual bool OnLeftMouseMoved     (Action* action, const StateEvent* stateEvent);
  virtual bool OnLeftMouseReleased  (Action* action, const StateEvent* stateEvent);
  virtual bool OnMiddleMousePressed (Action* action, const StateEvent* stateEvent);
  virtual bool OnMiddleMouseMoved   (Action* action, const StateEvent* stateEvent);
  virtual bool OnMiddleMouseReleased(Action* action, const StateEvent* stateEvent);
  virtual bool OnRightMousePressed  (Action* action, const StateEvent* stateEvent);
  virtual bool OnRightMouseMoved    (Action* action, const StateEvent* stateEvent);
  virtual bool OnRightMouseReleased (Action* action, const StateEvent* stateEvent);

protected:

  MIDASPaintbrushTool();          // purposely hidden
  virtual ~MIDASPaintbrushTool(); // purposely hidden

  /**
  \brief Called when the tool gets activated (registered to mitk::GlobalInteraction).

  Derived tools should call their parents implementation.
  */
  virtual void Activated();

  /**
  \brief Called when the tool gets deactivated (unregistered from mitk::GlobalInteraction).

  Derived tools should call their parents implementation.
  */
  virtual void Deactivated();

private:

  // Operation constant, used in Undo/Redo framework.
  static const mitk::OperationType MIDAS_PAINTBRUSH_TOOL_OP_EDIT_IMAGE;

  ///
  /// \brief Used for working out which voxels to edit.
  ///
  /// Essentially, we take two points, currentPoint and previousPoint in millimetre space
  /// and step along a line between them. At each step we convert from millimetres to voxels,
  /// and that list of voxels is the affected region.
  void GetListOfAffectedVoxels(
      const PlaneGeometry& planeGeometry,
      Point3D& currentPoint,
      Point3D& previousPoint,
      ProcessorType &processor);

  /// \brief Marks the initial mouse position when any of the left/middle/right mouse buttons are pressed.
  bool MarkInitialPosition(unsigned int imageNumber, Action* action, const StateEvent* stateEvent);

  /// \brief Sets an invalid region (indicating that we are not editing) on the chosen image number data node.
  void SetInvalidRegion(unsigned int imageNumber);

  /// \brief Sets a valid region property, taken from the bounding box of edited voxels, indicating that we are editing the given image number.
  void SetValidRegion(unsigned int imageNumber, std::vector<int>& boundingBox);

  /// \brief Method that actually sets the region property on a working image.
  void SetRegion(unsigned int imageNumber, bool valid, std::vector<int>& boundingBox);

  /// \brief Does the main functionality when the mouse moves.
  bool DoMouseMoved(Action* action,
      const StateEvent* stateEvent,
      int imageNumber,
      unsigned char valueForRedo,
      unsigned char valueForUndo
      );

  /// \brief Using the MITK to ITK access functions to run the ITK processor object.
  template<typename TPixel, unsigned int VImageDimension>
  void RunITKProcessor(
      itk::Image<TPixel, VImageDimension>* itkImage,
      ProcessorType::Pointer processor,
      bool redo,
      unsigned char valueToWrite
      );

  // Pointer to interface object, used as callback in Undo/Redo framework
  MIDASPaintbrushToolEventInterface::Pointer m_Interface;

  /// \brief Calculates the current image number.
  int GetImageNumber(bool isLeftMouseButton);

  // Cursor size for editing, and cursor type is currently always a cross.
  int m_CursorSize;

  // This is the 3D geometry associated with the m_WorkingImage, where we assume both working images have same size and geometry.
  mitk::Geometry3D* m_WorkingImageGeometry;

  // This points to the current working image, assuming that we are only ever processing, left, middle or right mouse button at any one time.
  mitk::Image* m_WorkingImage;

  // Used between MouseDown and MouseMoved events to track movement.
  mitk::Point3D m_MostRecentPointInMillimetres;

  // If m_ErosionMode is true, we update WorkingData 0 and 1, if m_ErosionMode is false, we update WorkingData 2 and 3.
  bool m_ErosionMode;

};//class

}//namespace

#endif
