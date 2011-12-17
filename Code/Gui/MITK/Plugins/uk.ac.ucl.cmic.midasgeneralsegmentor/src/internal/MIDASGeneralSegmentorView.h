/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-18 09:05:48 +0000 (Fri, 18 Nov 2011) $
 Revision          : $Revision: 7804 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef _MIDASGENERALSEGMENTORVIEW_H_INCLUDED
#define _MIDASGENERALSEGMENTORVIEW_H_INCLUDED

#include "ui_MIDASGeneralSegmentorViewControls.h"

#include <QString>
#include "MIDASGeneralSegmentorViewControlsWidget.h"
#include "MIDASGeneralSegmentorCommands.h"
#include "MIDASGeneralSegmentorViewPipeline.h"
#include "MIDASGeneralSegmentorViewHelper.h"
#include "QmitkMIDASBaseSegmentationFunctionality.h"
#include "mitkPointSet.h"
#include "mitkMIDASContourTool.h"
#include "mitkMIDASDrawTool.h"
#include "mitkMIDASPolyTool.h"
#include "mitkOperationActor.h"
#include "mitkOperation.h"
#include "itkImage.h"
#include "itkPointSet.h"
#include "itkIndex.h"
#include "itkContinuousIndex.h"
#include "itkMIDASHelper.h"
#include "itkMIDASRegionGrowingProcessor.h"
#include "itkMIDASRegionOfInterestCalculator.h"

class QButtonGroup;
class QGridLayout;

/**
 * \class MIDASGeneralSegmentorView
 * \brief View component for the CMIC MIDAS General Segmentor, providing functionality for hippocampus/ventricle segmentation.
 * \ingroup uk_ac_ucl_cmic_midasgeneralsegmentor_internal
 * \sa QmitkMIDASBaseSegmentationFunctionality
 */
class MIDASGeneralSegmentorView : public QmitkMIDASBaseSegmentationFunctionality, public mitk::OperationActor
{

  // this is needed for all Qt objects that should have a MOC object (everything that derives from QObject)
  Q_OBJECT

public:

  /**
   * \ingroup uk_ac_ucl_cmic_midasgeneralsegmentor_internal
   *
   * \class MIDASGeneralSegmentorViewEventInterface
   * \brief Interface class, simply to callback onto this class for Undo/Redo purposes.
   */
  class MIDASGeneralSegmentorViewEventInterface: public itk::Object, public mitk::OperationActor
  {
  public:
    MIDASGeneralSegmentorViewEventInterface() {};
    ~MIDASGeneralSegmentorViewEventInterface() {};
    void SetMIDASGeneralSegmentorView( MIDASGeneralSegmentorView* view )
    {
      m_View = view;
    }
    virtual void  ExecuteOperation(mitk::Operation* op)
    {
      m_View->ExecuteOperation(op);
    }
  private:
    MIDASGeneralSegmentorView* m_View;
  };

  MIDASGeneralSegmentorView();
  MIDASGeneralSegmentorView(const MIDASGeneralSegmentorView& other);
  virtual ~MIDASGeneralSegmentorView();

  // Each View for a plugin has its own globally unique ID.
  static const std::string VIEW_ID;
  virtual std::string GetViewID() const;

  /// \brief QmitkFunctionality's activate.
  virtual void Activated();

  /// \brief QmitkFunctionality's deactivate.
  virtual void Deactivated();

  /// \brief Method to enable this class to interact with the Undo/Redo framework.
  virtual void ExecuteOperation(mitk::Operation* operation);

protected slots:
 
  /// \brief Called when the user hits the button "New segmentation".
  virtual mitk::DataNode* OnCreateNewSegmentationButtonPressed();

  /// \brief Called when a segmentation tool is activated.
  void OnManualToolSelected(int id);

  void OnCleanButtonPressed();
  void OnWipeButtonPressed();
  void OnWipePlusButtonPressed();
  void OnWipeMinusButtonPressed();
  void OnOKButtonPressed();
  void OnResetButtonPressed();
  void OnCancelButtonPressed();
  void OnThresholdApplyButtonPressed();
  void OnRetainMarksCheckBoxToggled(bool b);
  void OnThresholdCheckBoxToggled(bool b);
  void OnSeePriorCheckBoxToggled(bool b);
  void OnSeeNextCheckBoxToggled(bool b);
  void OnPropagateUpButtonPressed();
  void OnPropagateDownButtonPressed();
  void OnPropagate3DButtonPressed();
  void OnLowerThresholdValueChanged(double d);
  void OnUpperThresholdValueChanged(double d);
  void OnOrientationAxialToggled(bool b);
  void OnOrientationSagittalToggled(bool b);
  void OnOrientationCoronalToggled(bool b);
  void OnSliceNumberChanged(int before, int after);

protected:

  /// \brief Returns the tool manager associated with this object.
  virtual mitk::ToolManager* GetToolManager();

  ///  \brief method to enable derived classes to turn widgets off/on
  virtual void EnableSegmentationWidgets(bool b);

  /// \brief This method creates all the controls for this view.
  virtual void CreateQtPartControl(QWidget *parent);

  /// \brief Creation of the connections of widgets and this class.
  virtual void CreateConnections();

  /// \brief Called when a node changed.
  virtual void NodeChanged(const mitk::DataNode* node);

  /// \brief For Irregular Volume Editing, a Segmentation image should have a grey scale parent, and 3 binary children called mitk::MIDASTool::REGION_GROWING_IMAGE_NAME and mitk::MIDASTool::SEE_PRIOR_IMAGE_NAME, and mitk::MIDASTool::SEE_NEXT_IMAGE_NAME,
  virtual bool IsNodeASegmentationImage(const mitk::DataNode::Pointer node);

  /// \brief We return false in all cases.
  virtual bool CanStartSegmentationForBinaryNode(const mitk::DataNode::Pointer node);

private:

  // Operation constants, used in Undo/Redo framework
  static const mitk::OperationType OP_THRESHOLD_APPLY;
  static const mitk::OperationType OP_PROPAGATE_UP;
  static const mitk::OperationType OP_PROPAGATE_DOWN;
  static const mitk::OperationType OP_WIPE_SLICE;
  static const mitk::OperationType OP_WIPE_PLUS;
  static const mitk::OperationType OP_WIPE_MINUS;
  static const mitk::OperationType OP_RETAIN_MARKS;

  /// \brief Used to create 3 helper images, one for work in progress (before 'accept' button clicked), one for 'see prior', one for 'see next'.
  mitk::DataNode::Pointer CreateHelperImage(mitk::Image::Pointer referenceImage, mitk::DataNode::Pointer segmentationNode, float r, float g, float b, std::string name);

  /// \brief Gets the orientation enum.
  itk::ORIENTATION_ENUM GetOrientationAsEnum();

  /// \brief Orientation selected, enable widgets etc.
  void OnOrientationSelected(itk::ORIENTATION_ENUM orientation);

  /// \brief Works out the current slice number that we are segmenting.
  int GetSliceNumber();

  /// \brief Works out the axis index number for the orientation we are segmenting.
  int GetAxis();

  /// \brief Looks for the Seeds stored in the data storage.
  mitk::PointSet* GetSeeds();

  /// \brief Helper method to toggle "see prior", and "see next".
  void SetVisiblityOnDerivedImage(std::string name, bool visibility);

  /// \brief Retrieves the min and max of the image (stored in mitk::Image), and sets the sliders accordingly
  void RecalculateMinAndMaxOfImage();

  /// \brief For each seed in the list of seeds, converts to millimetre position, and looks up the pixel value at that location, and updates the min and max
  void RecalculateMinAndMaxOfSeedValues();

  /// \brief Recalculate the min and max of seed values using ITK
  template<typename TPixel, unsigned int VImageDimension>
  void RecalculateMinAndMaxOfSeedValuesUsingITK(
      itk::Image<TPixel, VImageDimension>* itkImage,
      mitk::PointSet* points,
      double &min,
      double &max
      );

  /// \brief Given the current threshold values on the upper and lower slider, and all the seeds and contours, will recalculate the thresholded region in this slice. If updateVolume, result is applied to output segmentation.
  void UpdateRegionGrowing();

  /// \brief Templated function that updates the ITK region growing pipeline.
  template<typename TPixel, unsigned int VImageDimension>
  void InvokeITKRegionGrowingPipeline(
      itk::Image<TPixel, VImageDimension> *itkImage,
      bool skipUpdate,
      mitk::PointSet &seeds,
      mitk::MIDASDrawTool &drawTool,
      mitk::MIDASPolyTool &polyTool,
      itk::ORIENTATION_ENUM orientation,
      int sliceNumber,
      int axis,
      double lowerThreshold,
      double upperThreshold,
      mitk::DataNode::Pointer &outputRegionGrowingNode,
      mitk::Image::Pointer &outputRegionGrowingImage
      );

  /// \brief Whenever called, will take the current slice, and work out prior and next, and update those volumes.
  void UpdatePriorAndNext();

  /// \brief Templated function that copies the next/prior slice into the see next/prior image, so it shows up in rendered view.
  template<typename TPixel, unsigned int VImageDimension>
  void CopySlice(
        itk::Image<TPixel, VImageDimension>* sourceImage,
        itk::Image<TPixel, VImageDimension>* destinationImage,
        itk::ORIENTATION_ENUM orientation,
        int sliceNumber,
        int axis,
        int directionOffset
      );

  /// \brief Clears both images of the working data.
  void ClearWorkingData();

  /// \brief Clears a single ITK image.
  template<typename TPixel, unsigned int VImageDimension>
  void ClearITKImage(
      itk::Image<TPixel, VImageDimension>* itkImage
      );

  /// \brief Completely removes the current pipeline
  void DestroyPipeline();

  /// \brief Completely removes the current pipeline
  template<typename TPixel, unsigned int VImageDimension>
  void DestroyITKPipeline(
      itk::Image<TPixel, VImageDimension>* itkImage
      );

  /// \brief Copies an image, assumes input and output already allocated and same size.
  template<typename TPixel, unsigned int VImageDimension>
  void CopyImage(
      itk::Image<TPixel, VImageDimension>* input,
      itk::Image<TPixel, VImageDimension>* output
      );

  /// \brief Removes the images we are using for editing during segmentation.
  void RemoveWorkingData();

  /// \brief Does propagate up/down.
  void DoPropagate(bool isUp);

  /// \brief For propagation, will create a suitable propagate processor, and initialize it.
  template<typename TPixel, unsigned int VImageDimension>
  void CreateAndPopulatePropagateProcessor(
      itk::Image<TPixel, VImageDimension> *greyScaleImage,
      mitk::DataNode* regionGrowingNode,
      mitk::DataNode* referenceNode,
      mitk::PointSet &seeds,
      mitk::MIDASDrawTool &drawTool,
      mitk::MIDASPolyTool &polyTool,
      int sliceNumber,
      itk::ORIENTATION_ENUM orientation,
      bool isUp,
      double lowerThreshold,
      double upperThreshold,
      mitk::OperationEvent*& event
      );

  /// \brief Runs the propagate processor.
  template <typename TGreyScalePixel, unsigned int VImageDimension>
  void RunPropagateProcessor(
      itk::Image<TGreyScalePixel, VImageDimension>* greyScaleImage,
      mitk::OpPropagate *op,
      mitk::DataNode* regionGrowingNode,
      bool redo,
      int sliceNumber,
      itk::ORIENTATION_ENUM orientation);

  // Pointer to interface object, used as callback in Undo/Redo framework
  MIDASGeneralSegmentorViewEventInterface *m_Interface;

  /// \brief We hold a Map, containing a key comprised of the "typename TPixel, unsigned int VImageDimension"
  // as a key, and the object containing the whole pipeline.
  typedef std::pair<std::string, GeneralSegmentorPipelineInterface*> StringAndPipelineInterfacePair;
  std::map<std::string, GeneralSegmentorPipelineInterface*> m_TypeToPipelineMap;

  /// \brief All the controls for the main view part.
  MIDASGeneralSegmentorViewControlsWidget* m_GeneralControls;

  /// \brief Used to put the base class widgets, and these widgets above in a common layout.
  QGridLayout *m_Layout;
  QWidget *m_ContainerForSelectorWidget;
  QWidget *m_ContainerForControlsWidget;

  /// \brief Used to enforce the axial, coronal, sagittal buttons into a group.
  QButtonGroup *m_OrientationButtons;
};
#endif // _MIDASGENERALSEGMENTORVIEW_H_INCLUDED
