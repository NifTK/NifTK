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
#include "mitkImage.h"
#include "itkTimeStamp.h"
#include "MorphologicalSegmentorPipelineParams.h"
#include "MorphologicalSegmentorPipelineInterface.h"
#include "MorphologicalSegmentorPipeline.h"
#include "MIDASMorphologicalSegmentorViewControlsImpl.h"

/**
 * \class MIDASMorphologicalSegmentorView
 * \brief Provides the MIDAS brain segmentation functionality developed at Dementia Research Centre.
 *
 * \ingroup uk_ac_ucl_cmic_midasmorphologicalsegmentor_internal
 *
 * This plugin implements the paper:
 *
 * "Interactive algorithms for the segmentation and quantification of 3-D MRI brain scans"
 * by P. A. Freeborough, N. C. Fox and R. I. Kitney, published in
 * Computer Methods and Programs in Biomedicine 53 (1997) 15-25.
 *
 * \sa QmitkMIDASBaseSegmentationFunctionality
 * \sa MorphologicalSegmentorPipeline
 * \sa MorphologicalSegmentorPipelineInterface
 * \sa MorphologicalSegmentorPipelineParams
 */
class MIDASMorphologicalSegmentorView : public QmitkMIDASBaseSegmentationFunctionality
{

  // this is needed for all Qt objects that should have a MOC object (everything that derives from QObject)
  Q_OBJECT

public:

  /// \brief Constructor, which does almost nothing, as most construction of the view is done in CreateQtPartControl().
  MIDASMorphologicalSegmentorView();

  /// \brief Copy constructor which will throw a runtime exception, as no-one should call it.
  MIDASMorphologicalSegmentorView(const MIDASMorphologicalSegmentorView& other);

  /// \brief Destructor, which cleans up the controls created in this view.
  virtual ~MIDASMorphologicalSegmentorView();

  /// \brief Each View for a plugin has its own globally unique ID, this one is "uk.ac.ucl.cmic.midasmorphologicalsegmentor".
  static const std::string VIEW_ID;

  /// \brief Returns VIEW_ID="uk.ac.ucl.cmic.midasmorphologicalsegmentor".
  virtual std::string GetViewID() const;

  /// \brief Invoked when the DataManager selection changed.
  virtual void OnSelectionChanged(std::vector<mitk::DataNode*> nodes);

protected slots:
 
  /// \brief Called when the user hits the button "New segmentation".
  virtual mitk::DataNode* OnCreateNewSegmentationButtonPressed();

  /// \brief Called from MIDASMorphologicalSegmentorViewControlsImpl when thresholding widgets changed.
  void OnThresholdingValuesChanged(double lowerThreshold, double upperThreshold, int axialSlicerNumber);

  /// \brief Called from MIDASMorphologicalSegmentorViewControlsImpl when erosion widgets changed.
  void OnErosionsValuesChanged(double upperThreshold, int numberOfErosions);

  /// \brief Called from MIDASMorphologicalSegmentorViewControlsImpl when dilation widgets changed.
  void OnDilationValuesChanged(double lowerPercentage, double upperPercentage, int numberOfDilations);

  /// \brief Called from MIDASMorphologicalSegmentorViewControlsImpl when rethresholding widgets changed.
  void OnRethresholdingValuesChanged(int boxSize);

  /// \brief Called from MIDASMorphologicalSegmentorViewControlsImpl when a tab changes.
  void OnTabChanged(int i);

  /// \brief Called from MIDASMorphologicalSegmentorViewControlsImpl when OK button is clicked, which should finalise / finish and accept the segmentation.
  void OnOKButtonClicked();

  /// \brief Called from MIDASMorphologicalSegmentorViewControlsImpl when Clear button is clicked, which means "back to start", like a "reset" button.
  void OnClearButtonClicked();

  /// \brief Called from MIDASMorphologicalSegmentorViewControlsImpl when cancel button is clicked, which should mean "throw away" / "abandon" current segmentation.
  void OnCancelButtonClicked();

  /// \brief Called from QmitkMIDASToolSelectorWidget when a tool changes.... where we may need to enable or disable the editors from moving/changing position, zoom, etc.
  void OnToolSelected(int);

protected:

  /// \brief For Morphological Editing, if the tool manager has the correct WorkingData registered, then the Segmentation node is the immediate binary parent image of either working node.
  virtual mitk::DataNode* GetSegmentationNodeUsingToolManager();

  /// \brief For Morphological Editing, if the tool manager has the correct WorkingData registered, then the Segmentation image is the immediate binary parent image of either working node.
  virtual mitk::Image* GetSegmentationImageUsingToolManager();

  /// \brief For Morphological Editing, a Segmentation image should have a grey scale parent, and two binary children called SUBTRACTIONS_IMAGE_NAME and ADDITIONS_IMAGE_NAME.
  virtual bool IsNodeASegmentationImage(const mitk::DataNode::Pointer node);

  /// \brief For Morphological Editing, a Working image should be called either SUBTRACTIONS_IMAGE_NAME and ADDITIONS_IMAGE_NAME, and have a binary image parent.
  virtual bool IsNodeAWorkingImage(const mitk::DataNode::Pointer node);

  /// \brief For any binary image, we return true if the property midas.morph.stage is present, and false otherwise.
  virtual bool CanStartSegmentationForBinaryNode(const mitk::DataNode::Pointer node);

  /// \brief Assumes input is a valid segmentation node, then searches for the derived children of the node, looking for binary images called SUBTRACTIONS_IMAGE_NAME and ADDITIONS_IMAGE_NAME. Returns empty list if both not found.
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

  // We store a name for subtractions image.
  static const std::string SUBTRACTIONS_IMAGE_NAME;

  // We store a name for additions image.
  static const std::string ADDITIONS_IMAGE_NAME;

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
      std::vector<int>& editingRegion,
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

  // All the controls for the main Morphological Editor view part.
  MIDASMorphologicalSegmentorViewControlsImpl* m_MorphologicalControls;

  /// \brief Used to put the base class widgets, and these widgets above in a common layout.
  QGridLayout *m_Layout;

  /// \brief Container for Selector Widget (see base class).
  QWidget *m_ContainerForSelectorWidget;

  /// \brief Container for Tool Widget (see base class).
  QWidget *m_ContainerForToolWidget;

  /// \brief Container for the Morphological Controls Widgets (see this class).
  QWidget *m_ContainerForControlsWidget;

  /// \brief The mitk::ToolManager is created in base class, but we request and store locally the Tools ID.
  int m_PaintbrushToolId;

};

#endif // _MIDASMORPHOLOGICALSEGMENTORVIEW_H_INCLUDED
