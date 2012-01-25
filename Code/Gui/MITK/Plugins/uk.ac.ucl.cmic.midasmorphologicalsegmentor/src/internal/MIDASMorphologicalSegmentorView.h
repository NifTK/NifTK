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

#ifndef _MIDASMORPHOLOGICALSEGMENTORVIEW_H_INCLUDED
#define _MIDASMORPHOLOGICALSEGMENTORVIEW_H_INCLUDED

#include "QmitkMIDASBaseSegmentationFunctionality.h"
#include "MIDASMorphologicalSegmentorViewControlsImpl.h"
#include "mitkTool.h"
#include "mitkToolManager.h"
#include "mitkImage.h"
#include "itkImage.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkMIDASMaskByRegionImageFilter.h"
#include "itkMIDASConditionalErosionFilter.h"
#include "itkMIDASConditionalDilationFilter.h"
#include "itkMIDASRethresholdingFilter.h"
#include "itkMIDASLargestConnectedComponentFilter.h"
#include "itkExcludeImageFilter.h"
#include "itkInjectSourceImageGreaterThanZeroIntoTargetImageFilter.h"
#include "itkExtractImageFilter.h"
#include "itkPasteImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkTimeStamp.h"
#include "MorphologicalSegmentorPipelineParams.h"
#include "itkMIDASMorphologicalSegmentorLargestConnectedComponentImageFilter.h"

/**
 * \class MorphologicalSegmentorPipelineInterface
 * \brief Abstract interface to plug ITK pipeline into MITK framework.
 * \ingroup uk_ac_ucl_cmic_midasmorphologicalsegmentor_internal
 */
class MorphologicalSegmentorPipelineInterface
{
public:
  virtual void SetParam(MorphologicalSegmentorPipelineParams& p) = 0;
  virtual void Update(bool editingImageBeingEdited, bool additionsImageBeingEdited, int *editingRegion) = 0;
};

/**
 * \class MorphologicalSegmentorPipeline
 * \brief Implementation of MorphologicalSegmentorPipelineInterface to hook into ITK pipeline.
 * \ingroup uk_ac_ucl_cmic_midasmorphologicalsegmentor_internal
 */
template<typename TPixel, unsigned int VImageDimension>
class MorphologicalSegmentorPipeline : public MorphologicalSegmentorPipelineInterface
{
public:

  typedef itk::Image<TPixel, VImageDimension> GreyScaleImageType;
  typedef itk::Image<mitk::Tool::DefaultSegmentationDataType, VImageDimension> SegmentationImageType;

  typedef itk::BinaryThresholdImageFilter<GreyScaleImageType, SegmentationImageType> ThresholdingFilterType;
  typedef itk::MIDASMaskByRegionImageFilter<SegmentationImageType, SegmentationImageType> MaskByRegionFilterType;
  typedef itk::MIDASMorphologicalSegmentorLargestConnectedComponentImageFilter<SegmentationImageType, SegmentationImageType> LargestConnectedComponentFilterType;
  typedef itk::MIDASConditionalErosionFilter<GreyScaleImageType, SegmentationImageType, SegmentationImageType> ErosionFilterType;
  typedef itk::MIDASConditionalDilationFilter<GreyScaleImageType, SegmentationImageType, SegmentationImageType> DilationFilterType;
  typedef itk::MIDASRethresholdingFilter<GreyScaleImageType, SegmentationImageType, SegmentationImageType> RethresholdingFilterType;
  typedef itk::InjectSourceImageGreaterThanZeroIntoTargetImageFilter<SegmentationImageType, SegmentationImageType, SegmentationImageType> OrImageFilterType;
  typedef itk::ExcludeImageFilter<SegmentationImageType, SegmentationImageType, SegmentationImageType> ExcludeImageFilterType;
  typedef itk::ExtractImageFilter<SegmentationImageType, SegmentationImageType> RegionOfInterestImageFilterType;
  typedef itk::PasteImageFilter<SegmentationImageType, SegmentationImageType> PasteImageFilterType;

  MorphologicalSegmentorPipeline();
  void SetParam(MorphologicalSegmentorPipelineParams& p);
  void Update(bool editingImageBeingEdited, bool additionsImageBeingEdited, int *editingRegion);
  typename SegmentationImageType::Pointer GetOutput(bool editingImageBeingEdited, bool additionsImageBeingEdited);

  mitk::Tool::DefaultSegmentationDataType m_ForegroundValue;
  mitk::Tool::DefaultSegmentationDataType m_BackgroundValue;

  int m_Stage;

  typename ThresholdingFilterType::Pointer                     m_ThresholdingFilter;
  typename MaskByRegionFilterType::Pointer                     m_EarlyMaskFilter;
  typename LargestConnectedComponentFilterType::Pointer        m_EarlyConnectedComponentFilter;
  typename ErosionFilterType::Pointer                          m_ErosionFilter;
  typename DilationFilterType::Pointer                         m_DilationFilter;
  typename RethresholdingFilterType::Pointer                   m_RethresholdingFilter;
  typename MaskByRegionFilterType::Pointer                     m_LateMaskFilter;
  typename OrImageFilterType::Pointer                          m_OrImageFilter;
  typename ExcludeImageFilterType::Pointer                     m_ExcludeImageFilter;
  typename LargestConnectedComponentFilterType::Pointer        m_LateConnectedComponentFilter;
};

/**
 * \class MIDASMorphologicalSegmentorView
 * \brief Provides the MIDAS brain segmentation functionality developed at Dementia Research Centre.
 * \ingroup uk_ac_ucl_cmic_midasmorphologicalsegmentor_internal
 *
 * This plugin implements the paper:
 * "Interactive algorithms for the segmentation and quantification of 3-D MRI brain scans"
 * by P. A. Freeborough, N. C. Fox and R. I. Kitney, published in
 * Computer Methods and Programs in Biomedicine 53 (1997) 15-25.
 *
 * \sa QmitkMIDASBaseSegmentationFunctionality
 */
class MIDASMorphologicalSegmentorView : public QmitkMIDASBaseSegmentationFunctionality
{

  // this is needed for all Qt objects that should have a MOC object (everything that derives from QObject)
  Q_OBJECT

  public:

  MIDASMorphologicalSegmentorView();
  MIDASMorphologicalSegmentorView(const MIDASMorphologicalSegmentorView& other);
  virtual ~MIDASMorphologicalSegmentorView();

  // Each View for a plugin has its own globally unique ID.
  static const std::string VIEW_ID;
  virtual std::string GetViewID() const;

  /// \brief QmitkFunctionality's activate.
  virtual void Activated();

  /// \brief QmitkFunctionality's deactivate.
  virtual void Deactivated();

  /// \brief Invoked when the DataManager selection changed.
  virtual void OnSelectionChanged(std::vector<mitk::DataNode*> nodes);

protected slots:
 
  /// \brief Called when the user hits the button "New segmentation".
  virtual mitk::DataNode* OnCreateNewSegmentationButtonPressed();
  void OnThresholdingValuesChanged(double lowerThreshold, double upperThreshold, int axialSlicerNumber);
  void OnErosionsValuesChanged(double upperThreshold, int numberOfErosions);
  void OnDilationValuesChanged(double lowerPercentage, double upperPercentage, int numberOfDilations);
  void OnRethresholdingValuesChanged(int boxSize);
  void OnTabChanged(int i);
  void OnCursorWidthChanged(int i);
  void OnOKButtonClicked();
  void OnClearButtonClicked();
  void OnCancelButtonClicked();

protected:

  /// \brief Returns the tool manager associated with this object, which is created within this class.
  virtual mitk::ToolManager* GetToolManager();

  /// \brief For Morphological Editing, if the tool manager has the correct WorkingData registered, then the Segmentation node is the immediate binary parent image of either working node.
  virtual mitk::DataNode* GetSegmentationNodeUsingToolManager();

  /// \brief For Morphological Editing, if the tool manager has the correct WorkingData registered, then the Segmentation node is the immediate binary parent image of either working node.
  virtual mitk::Image* GetSegmentationImageUsingToolManager();

  /// \brief For Morphological Editing, a Segmentation image should have a grey scale parent, and two binary children called mitk::MIDASTool::SUBTRACTIONS_IMAGE_NAME and mitk::MIDASTool::ADDITIONS_IMAGE_NAME.
  virtual bool IsNodeASegmentationImage(const mitk::DataNode::Pointer node);

  /// \brief For Morphological Editing, a Working image should be called either mitk::MIDASTool::SUBTRACTIONS_IMAGE_NAME and mitk::MIDASTool::ADDITIONS_IMAGE_NAME, and have a binary image parent.
  virtual bool IsNodeAWorkingImage(const mitk::DataNode::Pointer node);

  /// \brief For any binary image, we return true if the property midas.morph.stage is present, and false otherwise.
  virtual bool CanStartSegmentationForBinaryNode(const mitk::DataNode::Pointer node);

  /// \brief Assumes input is a valid segmentation node, then searches for the derived children of the node, looking for binary images called mitk::MIDASTool::SUBTRACTIONS_IMAGE_NAME and mitk::MIDASTool::ADDITIONS_IMAGE_NAME. Returns empty list if both not found.
  virtual mitk::ToolManager::DataVectorType GetWorkingNodesFromSegmentationNode(const mitk::DataNode::Pointer node);

  /// \brief Assumes input is a valid working node, then searches for a binary parent node, returns NULL if not found.
  virtual mitk::DataNode* GetSegmentationNodeFromWorkingNode(const mitk::DataNode::Pointer node);

  /// \brief method to enable derived classes to turn widgets off/on
  virtual void EnableSegmentationWidgets(bool b);

  /// \brief Creation of the connections of main and control widget
  virtual void CreateConnections();

  /// \brief This method creates all the controls for this view
  virtual void CreateQtPartControl(QWidget *parent);

  /// \brief Called when a node changed.
  virtual void NodeChanged(const mitk::DataNode* node);

private:

  /// Some static strings, to avoid repetition.
  static const std::string PROPERTY_MIDAS_MORPH_SEGMENTATION_FINISHED;

  /// \brief Calls update on the ITK pipeline using the MITK AccessByItk macros.
  void UpdateSegmentation();

  /// \brief Templated function that updates the pipeline.
  template<typename TPixel, unsigned int VImageDimension>
  void InvokeITKPipeline(
      itk::Image<TPixel, VImageDimension>* itkImage,
      mitk::Image::Pointer& edits,
      mitk::Image::Pointer& additions,
      MorphologicalSegmentorPipelineParams& params,
      bool editingImageBeingEdited,
      bool additionsImageBeingEdited,
      bool isRestarting,
      int *editingRegion,
      mitk::Image::Pointer& outputImage
      );

  /// \brief Copies the final image out of the pipeline, and then disconnects the pipeline to stop it updating.
  void FinalizeSegmentation();

  /// \brief ITK method that actually does the work of finalizing the pipeline.
  template<typename TPixel, unsigned int VImageDimension>
  void FinalizeITKPipeline(
      itk::Image<TPixel, VImageDimension>* itkImage,
      mitk::Image::Pointer& outputImage
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

  /// \brief Removes the images we are using for editing during connection breaker from the DataStorage
  void RemoveWorkingData();

  /// \brief Returns the axis (0,1,2) that corresponds to the Axial direction.
  int GetAxialAxis();

  /// \brief Sets the morphological controls to default values specified by reference image, like min/max intensity range, number of axial slices etc.
  void SetControlsByImageData();

  /// \brief Sets the morphological controls by the current property values
  void SetControlsByParameterValues();

  /// \brief Looks up the reference image, and sets default values.
  void SetDefaultParameterValuesFromReferenceImage();

  /// \brief Retrieves the parameter values from the DataStorage node of the SegmentationImage.
  void GetParameterValues(MorphologicalSegmentorPipelineParams& params);

  /// \brief We hold a Map, containing a key comprised of the "typename TPixel, unsigned int VImageDimension"
  // as a key, and the object containing the whole pipeline.
  typedef std::pair<std::string, MorphologicalSegmentorPipelineInterface*> StringAndPipelineInterfacePair;
  std::map<std::string, MorphologicalSegmentorPipelineInterface*> m_TypeToPipelineMap;

  // All the controls for the main view part.
  MIDASMorphologicalSegmentorViewControlsImpl* m_MorphologicalControls;

  /// \brief Used to put the base class widgets, and these widgets above in a common layout.
  QGridLayout *m_Layout;
  QWidget *m_ContainerForSelectorWidget;
  QWidget *m_ContainerForControlsWidget;

  /// \brief we use a tool manager to store references to working/reference data, and a single draw tool.
  mitk::ToolManager::Pointer m_ToolManager;
  int m_PaintbrushToolId;

};

#endif // _MIDASMORPHOLOGICALSEGMENTORVIEW_H_INCLUDED
