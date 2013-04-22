/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkAffineTransformer.h"
#include <mitkDataNode.h>
#include <mitkImageAccessByItk.h>
#include <mitkImageStatisticsHolder.h>
#include <mitkITKImageImport.h>

#include "mitkDataStorageUtils.h"

#include "ConversionUtils.h"

const std::string mitk::AffineTransformer::VIEW_ID                   = "uk.ac.ucl.cmic.affinetransformview";
const std::string mitk::AffineTransformer::INITIAL_TRANSFORM_KEY     = "niftk.initaltransform";
const std::string mitk::AffineTransformer::INCREMENTAL_TRANSFORM_KEY = "niftk.incrementaltransform";
const std::string mitk::AffineTransformer::PRELOADED_TRANSFORM_KEY   = "niftk.preloadedtransform";
const std::string mitk::AffineTransformer::DISPLAYED_TRANSFORM_KEY   = "niftk.displayedtransform";
const std::string mitk::AffineTransformer::DISPLAYED_PARAMETERS_KEY  = "niftk.displayedtransformparameters";

namespace mitk
{

//-----------------------------------------------------------------------------
AffineTransformer::AffineTransformer()
  : m_RotateAroundCenter(false),
    m_DataStorage(0),
    m_CurrentDataNode(0)
{
}


//-----------------------------------------------------------------------------
AffineTransformer::~AffineTransformer()
{
}


//-----------------------------------------------------------------------------
void AffineTransformer::SetDataStorage(mitk::DataStorage::Pointer dataStorage)
{
  this->m_DataStorage = dataStorage;
  this->Modified();
}


//-----------------------------------------------------------------------------
mitk::DataStorage::Pointer AffineTransformer::GetDataStorage() const
{
  return this->m_DataStorage;
}

//-----------------------------------------------------------------------------
mitk::AffineTransformParametersDataNodeProperty::Pointer
     AffineTransformer::GetCurrentTransformParameters() const
{
  mitk::AffineTransformParametersDataNodeProperty::Pointer affineTransformParametersProperty
    = dynamic_cast<mitk::AffineTransformParametersDataNodeProperty*>(m_CurrentDataNode->GetProperty(DISPLAYED_PARAMETERS_KEY.c_str()));

  return affineTransformParametersProperty;
}

//-----------------------------------------------------------------------------
vtkSmartPointer<vtkMatrix4x4> AffineTransformer::GetTransformMatrixFromNode(std::string which) const
{
  vtkSmartPointer<vtkMatrix4x4> transform = mitk::AffineTransformDataNodeProperty::LoadTransformFromNode(which, *(m_CurrentDataNode.GetPointer()));
  return transform;
}

//-----------------------------------------------------------------------------
vtkSmartPointer<vtkMatrix4x4> AffineTransformer::GetCurrentTransformMatrix() const
{
  vtkSmartPointer<vtkMatrix4x4> transform = this->ComputeTransformFromParameters();
  return transform;
}

//-----------------------------------------------------------------------------
void AffineTransformer::OnNodeChanged(mitk::DataNode::Pointer node)
{
  // First check if the input node isn't null
  if (node.IsNotNull())
  {
    // Store the current node as a member variable.
    m_CurrentDataNode = node;
  }
  else return;

  // Initialise the selected node.
  this->InitialiseNodeProperties(node);
  
  // Initialise all of the selected nodes children.
  mitk::DataStorage::SetOfObjects::ConstPointer children = this->GetDataStorage()->GetDerivations(node);
  for (unsigned int i = 0; i < children->Size(); i++)
  {
    this->InitialiseNodeProperties(children->GetElement(i));
  }

  // Initialise the centre of rotation member variable.
  typedef itk::Point<mitk::ScalarType, 3> PointType;
  PointType centrePoint = m_CurrentDataNode->GetData()->GetGeometry()->GetCenter();
  
  m_CentreOfRotation[0] = centrePoint[0];
  m_CentreOfRotation[1] = centrePoint[1];
  m_CentreOfRotation[2] = centrePoint[2];

  MITK_DEBUG << "OnSelectionChanged, set centre to " << m_CentreOfRotation[0] << ", " << m_CentreOfRotation[1] << ", " << m_CentreOfRotation[2] << std::endl;
}



//-----------------------------------------------------------------------------
void AffineTransformer::UpdateTransformationGeometry()
{
    /**************************************************************
   * This is the main method composing and calculating matrices.
   **************************************************************/
  if (m_CurrentDataNode.IsNotNull())
  {
    vtkSmartPointer<vtkMatrix4x4> transformDisplayed = mitk::AffineTransformDataNodeProperty::LoadTransformFromNode(DISPLAYED_TRANSFORM_KEY.c_str(), *(m_CurrentDataNode.GetPointer()));
    vtkSmartPointer<vtkMatrix4x4> transformPreLoaded = mitk::AffineTransformDataNodeProperty::LoadTransformFromNode(PRELOADED_TRANSFORM_KEY.c_str(), *(m_CurrentDataNode.GetPointer()));

    vtkSmartPointer<vtkMatrix4x4> invertedDisplayedTransform = vtkMatrix4x4::New();
    vtkMatrix4x4::Invert(transformDisplayed, invertedDisplayedTransform);

    vtkSmartPointer<vtkMatrix4x4> invertedTransformPreLoaded = vtkMatrix4x4::New();
    vtkMatrix4x4::Invert(transformPreLoaded, invertedTransformPreLoaded);

    vtkSmartPointer<vtkMatrix4x4> newTransformAccordingToParameters = this->ComputeTransformFromParameters();

    vtkSmartPointer<vtkMatrix4x4> invertedTransforms = vtkMatrix4x4::New();
    vtkSmartPointer<vtkMatrix4x4> transformsBeforeAffine = vtkMatrix4x4::New();
    vtkSmartPointer<vtkMatrix4x4> finalAffineTransform = vtkMatrix4x4::New();

    vtkMatrix4x4::Multiply4x4(invertedTransformPreLoaded, invertedDisplayedTransform, invertedTransforms);
    vtkMatrix4x4::Multiply4x4(transformPreLoaded, invertedTransforms, transformsBeforeAffine);
    vtkMatrix4x4::Multiply4x4(newTransformAccordingToParameters, transformsBeforeAffine, finalAffineTransform);

    this->UpdateNodeProperties(newTransformAccordingToParameters, finalAffineTransform, m_CurrentDataNode);

    mitk::DataStorage::SetOfObjects::ConstPointer children = this->GetDataStorage()->GetDerivations(m_CurrentDataNode);
    
    for (unsigned int i = 0; i < children->Size(); i++)
    {
      this->UpdateNodeProperties(
          newTransformAccordingToParameters,
          finalAffineTransform,
          children->GetElement(i)
          );
    }

    //QmitkAbstractView::RequestRenderWindowUpdate();
  }
}

vtkSmartPointer<vtkMatrix4x4> AffineTransformer::ComputeTransformFromParameters(void) const
{
  vtkSmartPointer<vtkMatrix4x4> sp_inc, sp_tmp, sp_swap;
  double incVals[4][4], partInc[4][4], result[4][4];
  int cInd;

  vtkMatrix4x4::Identity(&incVals[0][0]);

  if (m_RotateAroundCenter) 
  {
    MITK_DEBUG << "Transform applied wrt. ("
      << m_CentreOfRotation[0] << ", "
      << m_CentreOfRotation[1] << ", "
      << m_CentreOfRotation[2] << ")\n";

    for (cInd = 0; cInd < 3; cInd++)
      incVals[cInd][3] = -m_CentreOfRotation[cInd];
  }

  vtkMatrix4x4::Identity(&partInc[0][0]);
  partInc[0][0] = m_Scaling[0]/100.0;
  partInc[1][1] = m_Scaling[1]/100.0;
  partInc[2][2] = m_Scaling[2]/100.0;
  
  vtkMatrix4x4::Multiply4x4(&partInc[0][0], &incVals[0][0], &result[0][0]);
  std::copy(&result[0][0], &result[0][0] + 16, &incVals[0][0]);

  vtkMatrix4x4::Identity(&partInc[0][0]);
  partInc[0][1] = m_Shearing[0];
  partInc[0][2] = m_Shearing[1];
  partInc[1][2] = m_Shearing[2];
  
  vtkMatrix4x4::Multiply4x4(&partInc[0][0], &incVals[0][0], &result[0][0]);
  std::copy(&result[0][0], &result[0][0] + 16, &incVals[0][0]);

  {
    double calpha, salpha, alpha;

    alpha = NIFTK_PI*m_Rotation[0]/180;
    calpha = cos(alpha);
    salpha = sin(alpha);

    vtkMatrix4x4::Identity(&partInc[0][0]);
    partInc[1][1] = calpha;
    partInc[1][2] = salpha;
    partInc[2][1] = -salpha;
    partInc[2][2] = calpha;
    vtkMatrix4x4::Multiply4x4(&partInc[0][0], &incVals[0][0], &result[0][0]);

    alpha = NIFTK_PI*m_Rotation[1]/180.0;
    calpha = cos(alpha);
    salpha = sin(alpha);

    vtkMatrix4x4::Identity(&partInc[0][0]);
    partInc[0][0] = calpha;
    partInc[0][2] = salpha;
    partInc[2][0] = -salpha;
    partInc[2][2] = calpha;
    vtkMatrix4x4::Multiply4x4(&partInc[0][0], &result[0][0], &incVals[0][0]);

    alpha = NIFTK_PI*m_Rotation[2]/180.0;
    calpha = cos(alpha);
    salpha = sin(alpha);

    vtkMatrix4x4::Identity(&partInc[0][0]);
    partInc[0][0] = calpha;
    partInc[0][1] = salpha;
    partInc[1][0] = -salpha;
    partInc[1][1] = calpha;
    vtkMatrix4x4::Multiply4x4(&partInc[0][0], &incVals[0][0], &result[0][0]);

    std::copy(&result[0][0], &result[0][0] + 16, &incVals[0][0]);
  }

  incVals[0][3] += m_Translation[0];
  incVals[1][3] += m_Translation[1];
  incVals[2][3] += m_Translation[2];

  if (m_RotateAroundCenter) 
  {
    for (cInd = 0; cInd < 3; cInd++) incVals[cInd][3] += m_CentreOfRotation[cInd];
  }

  sp_inc = vtkSmartPointer<vtkMatrix4x4>::New();
  std::copy(&incVals[0][0], &incVals[0][0] + 4*4, &sp_inc->Element[0][0]);

  return sp_inc;
}


//-----------------------------------------------------------------------------
void AffineTransformer::FinalizeTransformation()
{
  //mitk::DataNode::Pointer workingDataNode = this->GetToolManager()->GetWorkingData(0);
  //if (workingDataNode.IsNotNull())
  //{
  //  mitk::DataNode::Pointer parent = mitk::FindFirstParentImage(this->GetDataStorage().GetPointer(), workingDataNode, true);
  //  if (parent.IsNotNull())
  //  {
  //    mitk::Image::Pointer outputImage = mitk::Image::New();
  //    mitk::Image::Pointer referenceImage = this->GetReferenceImageFromToolManager(0);

  //    try
  //    {
  //      AccessFixedDimensionByItk_n(referenceImage, FinalizeITKPipeline, 3, (outputImage));
  //    }
  //    catch(const mitk::AccessByItkException& e)
  //    {
  //      MITK_ERROR << "Caught exception, so finalize pipeline" << e.what();
  //    }
  //    this->RemoveWorkingData();
  //    this->DestroyPipeline();

  //    parent->SetData( outputImage );
  //    parent->ReplaceProperty(AffineTransformer::PROPERTY_MIDAS_MORPH_SEGMENTATION_FINISHED.c_str(), mitk::BoolProperty::New(true));

  //    UpdateVolumeProperty(outputImage, parent);
  //  }
  //}
}


//-----------------------------------------------------------------------------
void AffineTransformer::ClearWorkingData()
{
  //for (unsigned int i = 0; i < 4; i++)
  //{
  //  mitk::Image::Pointer image = this->GetWorkingImageFromToolManager(i);
  //  mitk::DataNode::Pointer node = this->GetToolManager()->GetWorkingData(i);

  //  if (image.IsNotNull() && node.IsNotNull())
  //  {
  //    try
  //    {
  //      AccessFixedDimensionByItk(image, ClearITKImage, 3);

  //      image->Modified();
  //      node->Modified();
  //    }
  //    catch(const mitk::AccessByItkException& e)
  //    {
  //      MITK_ERROR << "AffineTransformer::ClearWorkingData: i=" << i << ", caught exception, so abandoning clearing the segmentation image:" << e.what();
  //    }
  //  }
  //}
}


//-----------------------------------------------------------------------------
void AffineTransformer::RemoveWorkingData()
{
  //mitk::ToolManager* toolManager = this->GetToolManager();

  //mitk::ToolManager::DataVectorType workingData = toolManager->GetWorkingData();

  //for (unsigned int i = 0; i < workingData.size(); i++)
  //{
  //  mitk::DataNode* node = workingData[i];
  //  this->GetDataStorage()->Remove(node);
  //}

  //mitk::ToolManager::DataVectorType emptyWorkingDataArray;
  //toolManager->SetWorkingData(emptyWorkingDataArray);
  //toolManager->ActivateTool(-1);
}

/** \brief Slot for all changes to transformation parameters. */
void AffineTransformer::OnParametersChanged(mitk::AffineTransformParametersDataNodeProperty::Pointer paramsProperty)
{
  mitk::AffineTransformParametersDataNodeProperty::ParametersType params = paramsProperty->GetAffineTransformParameters();

  // Get rotation paramters first
  m_Rotation[0]    = params[0];
  m_Rotation[1]    = params[1];
  m_Rotation[2]    = params[2];

  // Get translation paramters
  m_Translation[0] = params[3];
  m_Translation[1] = params[4];
  m_Translation[2] = params[5];

  // Get scaling paramters
  m_Scaling[0]     = params[6];
  m_Scaling[1]     = params[7];
  m_Scaling[2]     = params[8];

  // Get shearing paramters
  m_Shearing[0]    = params[9];
  m_Shearing[1]    = params[10];
  m_Shearing[2]    = params[11];

  if (params[12] == 1)
  {
    m_RotateAroundCenter = true;
  }
  else
  {
    m_RotateAroundCenter = false;
  }
}

  /** \brief Slot for radio button state changes. */
  void AffineTransformer::OnParameterChanged(const bool)
  {
  }

  /** \brief Slot for reset button that resets the parameter controls, and updates node geometry accordingly. */
  void AffineTransformer::OnResetTransformPushed()
  {
  }

  /** \brief Slot for saving transform to disk. */
  void AffineTransformer::OnSaveTransformPushed()
  {
  }

  /** \brief Slot for loading transform from disk. */
  void AffineTransformer::OnLoadTransformPushed()
  {
  }

  /** \brief Slot for loading transform from disk. */
  void AffineTransformer::OnApplyTransformPushed()
  {
  }

  /** \brief Slot for resampling the current image. */
  void AffineTransformer::OnResampleTransformPushed()
  {
  }

/** Called by _InitialiseNodeProperties to initialise (to Identity) a specified transform property on a node. */
void AffineTransformer::InitialiseTransformProperty(std::string name, mitk::DataNode::Pointer node)
{
  mitk::AffineTransformDataNodeProperty::Pointer transform
    = dynamic_cast<mitk::AffineTransformDataNodeProperty*>(node->GetProperty(name.c_str()));

  if (transform.IsNull())
  {
    transform = mitk::AffineTransformDataNodeProperty::New();
    transform->Identity();
    node->SetProperty(name.c_str(), transform);
  }
}

/** Called by OnSelectionChanged to setup a node with default transformation properties, if it doesn't already have them. */
void AffineTransformer::InitialiseNodeProperties(mitk::DataNode::Pointer node)
{
  // Make sure the node has the specified properties listed below, and if not create defaults.
  InitialiseTransformProperty(INCREMENTAL_TRANSFORM_KEY, node);
  InitialiseTransformProperty(PRELOADED_TRANSFORM_KEY, node);
  InitialiseTransformProperty(DISPLAYED_TRANSFORM_KEY, node);

  mitk::AffineTransformParametersDataNodeProperty::Pointer affineTransformParametersProperty
    = dynamic_cast<mitk::AffineTransformParametersDataNodeProperty*>(node->GetProperty(DISPLAYED_PARAMETERS_KEY.c_str()));
  
  if (affineTransformParametersProperty.IsNull())
  {
    affineTransformParametersProperty = mitk::AffineTransformParametersDataNodeProperty::New();
    affineTransformParametersProperty->Identity();
    node->SetProperty(DISPLAYED_PARAMETERS_KEY.c_str(), affineTransformParametersProperty);
  }

  // In addition, if we have not already done so, we take any existing geometry,
  // and store it back on the node as the "Initial" geometry.
  mitk::AffineTransformDataNodeProperty::Pointer transform
    = dynamic_cast<mitk::AffineTransformDataNodeProperty*>(node->GetProperty(INITIAL_TRANSFORM_KEY.c_str()));
  
  if (transform.IsNull())
  {
    transform = mitk::AffineTransformDataNodeProperty::New();
    transform->SetTransform(*(const_cast<const vtkMatrix4x4*>(node->GetData()->GetGeometry()->GetVtkTransform()->GetMatrix())));
    node->SetProperty(INITIAL_TRANSFORM_KEY.c_str(), transform);
  }

  mitk::BaseData *data = node->GetData();
  if ( data == NULL )
  {
    MITK_ERROR << "No data object present!";
  }
}


  /** Called by _UpdateTransformationGeometry to set new transformations in the right properties of the node. */
  void AffineTransformer::UpdateNodeProperties(const vtkSmartPointer<vtkMatrix4x4> displayedTransformFromParameters,
                            const vtkSmartPointer<vtkMatrix4x4> incrementalTransformToBeComposed,
                            mitk::DataNode::Pointer)
  {
  }

  /** Called by _UpdateNodeProperties to update a transform property on a given node. */
  void AffineTransformer::UpdateTransformProperty(std::string name, vtkSmartPointer<vtkMatrix4x4> transform, mitk::DataNode& node)
  {
  }

  /** The transform loaded from file is applied to the current node, and all its children, and it resets the GUI parameters to Identity, and hence the DISPLAY_TRANSFORM and DISPLAY_PARAMETERS to Identity.*/
  void AffineTransformer::ApplyLoadedTransformToNode(const vtkSmartPointer<vtkMatrix4x4> transformFromFile, mitk::DataNode& node)
  {
  }

  /** \brief Updates the transform on the current node, and it's children. */
  void AffineTransformer::UpdateTransformationGeometry()
  {
  }

  /** \brief Applies a re-sampling to the current node. */
  void AffineTransformer::ApplyResampleToCurrentNode()
  {
  }



} // end namespace
