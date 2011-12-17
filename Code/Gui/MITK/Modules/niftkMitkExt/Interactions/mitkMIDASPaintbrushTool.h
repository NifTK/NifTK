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
#include "mitkMIDASContourTool.h"
#include "mitkOperationActor.h"
#include "mitkOperation.h"
#include "itkImage.h"
#include "itkImageUpdatePixelWiseSingleValueProcessor.h"

namespace mitk
{

/**
 * \class OpEditImage
 * \brief Nested class to hold data to pass back to this MIDASPaintbrushTool,
 * so that this MIDASPaintbrushTool can execute the Undo/Redo command.
 */
class OpEditImage: public mitk::Operation
{
public:
  typedef itk::ImageUpdatePixelWiseSingleValueProcessor<mitk::Tool::DefaultSegmentationDataType, 3> ProcessorType;

  OpEditImage(
      mitk::OperationType type,
      bool redo,
      int imageNumber,
      unsigned char valueToWrite,
      mitk::Image* imageToEdit,
      mitk::DataNode* nodeToEdit,
      ProcessorType* processor
      );
  ~OpEditImage() {};
  bool IsRedo() const { return m_Redo; }
  int GetImageNumber() const { return m_ImageNumber; }
  unsigned char GetValueToWrite() const { return m_ValueToWrite; }
  mitk::Image* GetImageToEdit() const { return m_ImageToEdit; }
  mitk::DataNode* GetNodeToEdit() const { return m_NodeToEdit; }
  ProcessorType::Pointer GetProcessor() const { return m_Processor; }

private:
  bool m_Redo;
  int m_ImageNumber;
  unsigned char m_ValueToWrite;
  mitk::Image* m_ImageToEdit;
  mitk::DataNode* m_NodeToEdit;
  ProcessorType::Pointer m_Processor;
};

 /**
  * \class MIDASPaintbrushTool
  * \brief MIDAS paint brush tool used during editing of the morphological editor.
  *
  * Note the following:
  * <pre>
  * 1.) Writes into 2 images, so ToolManager must have 2 volume to edit into.
  * 2.) We derive from MIDASContourTool mainly for consistency as we use the
  * methods that setup geometry, so we know which slice we are affecting.
  * 3.) Derives from mitk::OperationActor, so this tool supports undo/redo.
  * </pre>
  */
class NIFTKMITKEXT_EXPORT MIDASPaintbrushTool : public MIDASContourTool
{
  typedef itk::Image<mitk::Tool::DefaultSegmentationDataType, 3> ImageType;
  typedef itk::ImageUpdatePixelWiseSingleValueProcessor<mitk::Tool::DefaultSegmentationDataType, 3> ProcessorType;

  /**
   * \class MIDASPaintbrushToolEventInterface
   * \brief Interface class, simply to callback onto this class.
   */
  class MIDASPaintbrushToolEventInterface: public itk::Object, public mitk::OperationActor
  {
  public:
    MIDASPaintbrushToolEventInterface() {};
    ~MIDASPaintbrushToolEventInterface() {};
    void SetMIDASPaintbrushTool( MIDASPaintbrushTool* paintbrushTool )
    {
      m_MIDASPaintBrushTool = paintbrushTool;
    }
    virtual void  ExecuteOperation(mitk::Operation* op)
    {
      m_MIDASPaintBrushTool->ExecuteOperation(op);
    }
  private:
    MIDASPaintbrushTool* m_MIDASPaintBrushTool;
  };

  public:
  mitkClassMacro(MIDASPaintbrushTool, MIDASPaintbrushTool);
  itkNewMacro(MIDASPaintbrushTool);

  /** Strings to help the tool identify itself in GUI. */
  virtual const char* GetName() const;
  virtual const char** GetXPM() const;

  // We store a string of a property to say we are editing.
  static const std::string EDITING_PROPERTY_NAME;

  // Properties to hold the edited region
  static const std::string EDITING_PROPERTY_INDEX_X;
  static const std::string EDITING_PROPERTY_INDEX_Y;
  static const std::string EDITING_PROPERTY_INDEX_Z;
  static const std::string EDITING_PROPERTY_SIZE_X;
  static const std::string EDITING_PROPERTY_SIZE_Y;
  static const std::string EDITING_PROPERTY_SIZE_Z;
  static const std::string EDITING_PROPERTY_REGION_SET;

  /** Method to enable this class to interact with the Undo/Redo framework. */
  virtual void ExecuteOperation(Operation* operation);

  // We essentially need 3 mouse buttons
  // Left = add to additional volume that gets added to segmented volume.
  // Middle = add to subtracting volume, which affects connection breaker.
  // Right = subtract from subtraction volume, which affects connection breaker.
  // But, the actual mouse config may be different on different Operating Systems.
  virtual bool OnLeftMousePressed (Action* action, const StateEvent* stateEvent);
  virtual bool OnLeftMouseMoved   (Action* action, const StateEvent* stateEvent);
  virtual bool OnLeftMouseReleased(Action* action, const StateEvent* stateEvent);
  virtual bool OnMiddleMousePressed (Action* action, const StateEvent* stateEvent);
  virtual bool OnMiddleMouseMoved   (Action* action, const StateEvent* stateEvent);
  virtual bool OnMiddleMouseReleased(Action* action, const StateEvent* stateEvent);
  virtual bool OnRightMousePressed (Action* action, const StateEvent* stateEvent);
  virtual bool OnRightMouseMoved   (Action* action, const StateEvent* stateEvent);
  virtual bool OnRightMouseReleased(Action* action, const StateEvent* stateEvent);

  /** Set/Get methods to set the Cursor width. Default 1. */
  itkSetMacro(CursorSize, int);
  itkGetConstMacro(CursorSize, int);

protected:

  MIDASPaintbrushTool(); // purposely hidden
  virtual ~MIDASPaintbrushTool(); // purposely hidden

private:

  // Operation constant, used in Undo/Redo framework
  static const mitk::OperationType OP_EDIT_IMAGE;

  // Pointer to interface object, used as callback in Undo/Redo framework
  MIDASPaintbrushToolEventInterface *m_Interface;

  // Used between MouseDown and MouseMoved events to track movement.
  mitk::Point3D m_MostRecentPointInMillimetres;

  // Cursor size for editing
  int m_CursorSize;

  // Used for working out which voxels to edit.
  void GetListOfAffectedVoxels(
      const PlaneGeometry& planeGeometry,
      Point3D& currentPoint,
      Point3D& previousPoint,
      ProcessorType &processor);

  // Marks the initial mouse position when any of the mouse buttons are pressed.
  bool MarkInitialPosition(Action* action, const StateEvent* stateEvent);

  // Does the main functionality when the mouse moves, calls DoInitialCheck, then calls EditVoxels to write the specified valueToWrite into the specified imageNumber
  bool DoMouseMoved(Action* action,
      const StateEvent* stateEvent,
      int imageNumber,
      unsigned char valueForRedo,
      unsigned char valueForUndo
      );

  // To perform any logic before any of the MouseMoved methods.
  bool DoInitialCheck(Action* action, const StateEvent* stateEvent);

  // Tags a specified image with a boolean indicating whether the region is up to date.
  void UpdateRegionSetProperty(int imageNumber, bool isRegionSet);

  // Tags a specified image with a boolean to indicate if we are currently editing it.
  void UpdateEditingProperty(int imageNumber, bool editingPropertyValue);

  // Using the MITK to ITK access functions to run my ITK processor object.
  template<typename TPixel, unsigned int VImageDimension>
  void RunITKProcessor(
      itk::Image<TPixel, VImageDimension>* itkImage,
      ProcessorType::Pointer processor,
      bool redo,
      unsigned char valueToWrite
      );

};//class

}//namespace

#endif
