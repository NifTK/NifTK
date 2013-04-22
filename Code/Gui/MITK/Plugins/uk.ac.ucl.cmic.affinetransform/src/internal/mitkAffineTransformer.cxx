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

#include <itkResampleImageFilter.h>
#include <itkVectorResampleImageFilter.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkVectorLinearInterpolateImageFunction.h>
#include <itkAffineTransform.h>
#include <itkImage.h>
#include <itkImageFileWriter.h>
#include <itkRGBAPixel.h>
#include <itkRGBPixel.h>
#include <itkTransformFileReader.h>
#include <itkTransformFileWriter.h>


// STL
#include <cmath>
#include <algorithm>
#include <cassert>

#include <QString>

const std::string mitk::AffineTransformer::VIEW_ID                   = "uk.ac.ucl.cmic.affinetransformview";
const std::string mitk::AffineTransformer::INITIAL_TRANSFORM_KEY     = "niftk.initaltransform";
const std::string mitk::AffineTransformer::INCREMENTAL_TRANSFORM_KEY = "niftk.incrementaltransform";
const std::string mitk::AffineTransformer::PRELOADED_TRANSFORM_KEY   = "niftk.preloadedtransform";
const std::string mitk::AffineTransformer::DISPLAYED_TRANSFORM_KEY   = "niftk.displayedtransform";
const std::string mitk::AffineTransformer::DISPLAYED_PARAMETERS_KEY  = "niftk.displayedtransformparameters";

//-----------------------------------------------------------------------------
//              Templated ITK pipelines to perform the resampling
//-----------------------------------------------------------------------------
namespace
{

//-----------------------------------------------------------------------------
template <typename TPixelType, unsigned int t_Dim>
void _ApplyTransform(itk::Image<TPixelType, t_Dim> *p_itkImg, const vtkMatrix4x4 &transform) 
{
  typedef itk::Image<TPixelType, t_Dim> __ITKImageType;

  mitk::Image::Pointer sp_transImg;
  typename itk::ResampleImageFilter<__ITKImageType, __ITKImageType>::Pointer sp_resampler;
  typename itk::AffineTransform<double, t_Dim>::Pointer sp_itkTransform;

  sp_resampler = itk::ResampleImageFilter<__ITKImageType, __ITKImageType>::New();
  sp_itkTransform = _ConvertToITKTransform<t_Dim, true>(transform);
  sp_resampler->SetTransform(sp_itkTransform);

  sp_resampler->SetInput(p_itkImg);
  sp_resampler->SetInterpolator(itk::LinearInterpolateImageFunction<__ITKImageType>::New());

  sp_resampler->SetUseReferenceImage(true);
  sp_resampler->SetReferenceImage(p_itkImg);

  try {
    sp_resampler->UpdateLargestPossibleRegion();

    typename itk::ImageRegionConstIterator<__ITKImageType> resampledIterator(sp_resampler->GetOutput(), sp_resampler->GetOutput()->GetLargestPossibleRegion());
    typename itk::ImageRegionIterator<__ITKImageType> inputIterator(p_itkImg, sp_resampler->GetOutput()->GetLargestPossibleRegion());
    for (resampledIterator.GoToBegin(), inputIterator.GoToBegin();
      !resampledIterator.IsAtEnd();
      ++resampledIterator, ++inputIterator)
    {
      inputIterator.Set(resampledIterator.Get());
    }
  } catch (itk::ExceptionObject &r_itkEx) {
    MITK_ERROR << r_itkEx.what() << std::endl;
    return;
  }

  MITK_DEBUG << "Processing: success\n";
}

//-----------------------------------------------------------------------------
template <typename TMultiChannelPixelType, unsigned int t_Dim>
void _ApplyTransformMultiChannel(itk::Image<TMultiChannelPixelType, t_Dim> *p_itkImg, const vtkMatrix4x4 &transform) 
{
  typedef itk::Image<TMultiChannelPixelType, t_Dim> __ITKImageType;

  mitk::Image::Pointer sp_transImg;
  typename itk::VectorResampleImageFilter<__ITKImageType, __ITKImageType>::Pointer sp_resampler;
  typename itk::AffineTransform<double, t_Dim>::Pointer sp_itkTransform;

  sp_resampler = itk::VectorResampleImageFilter<__ITKImageType, __ITKImageType>::New();
  sp_itkTransform = _ConvertToITKTransform<t_Dim, true>(transform);
  sp_resampler->SetTransform(sp_itkTransform);

  sp_resampler->SetInput(p_itkImg);
  sp_resampler->SetInterpolator(itk::VectorLinearInterpolateImageFunction<__ITKImageType>::New());

  sp_resampler->SetSize(p_itkImg->GetLargestPossibleRegion().GetSize());
  sp_resampler->SetOutputSpacing(p_itkImg->GetSpacing());
  sp_resampler->SetOutputDirection(p_itkImg->GetDirection());
  sp_resampler->SetOutputOrigin(p_itkImg->GetOrigin());

  try {
    sp_resampler->UpdateLargestPossibleRegion();

    typename itk::ImageRegionConstIterator<__ITKImageType> resampledIterator(sp_resampler->GetOutput(), sp_resampler->GetOutput()->GetLargestPossibleRegion());
    typename itk::ImageRegionIterator<__ITKImageType> inputIterator(p_itkImg, sp_resampler->GetOutput()->GetLargestPossibleRegion());
    for (resampledIterator.GoToBegin(), inputIterator.GoToBegin();
      !resampledIterator.IsAtEnd();
      ++resampledIterator, ++inputIterator)
    {
      inputIterator.Set(resampledIterator.Get());
    }

  } catch (itk::ExceptionObject &r_itkEx) {
    MITK_ERROR << r_itkEx.what() << std::endl;
    return;
  }

  MITK_DEBUG << "Processing: success\n";
}

//-----------------------------------------------------------------------------
template <const unsigned int t_Dim, const bool t_DoInvert>
static typename itk::AffineTransform<double, t_Dim>::Pointer _ConvertToITKTransform(const vtkMatrix4x4 &transform)
{
  typedef itk::AffineTransform<double, t_Dim> __ITKTransform;

  typename __ITKTransform::Pointer sp_itkTransform;
  typename __ITKTransform::MatrixType itkMatrix;
  typename __ITKTransform::OutputVectorType itkVec;
  unsigned int rInd, cInd;
  vtkSmartPointer<vtkMatrix4x4> sp_vtkInptTransform;

  /*
  * -Invert input transform since itk resampler expects an inverse transform that can be applied to the coordinates.
  */
  sp_itkTransform = __ITKTransform::New();
  sp_vtkInptTransform =vtkSmartPointer<vtkMatrix4x4>::New();

  if (t_DoInvert) 
  {
    vtkMatrix4x4::Invert(const_cast<vtkMatrix4x4*>(&transform), sp_vtkInptTransform);
  } 
  else 
  {
    sp_vtkInptTransform->DeepCopy(const_cast<vtkMatrix4x4*>(&transform));
  }

  MITK_DEBUG << "Converting transform " << std::endl;
  //sp_vtkInptTransform->PrintSelf(MITK_DEBUG, *vtkIndent::New());
  MITK_DEBUG << "to ITK\n";

  for (rInd = 0; rInd < t_Dim; rInd++) 
  {
    for (cInd = 0; cInd < t_Dim; cInd++) 
      itkMatrix(rInd,cInd) = sp_vtkInptTransform->Element[rInd][cInd];

    itkVec[rInd] = sp_vtkInptTransform->Element[rInd][3];
  }

  sp_itkTransform->SetMatrix(itkMatrix);
  sp_itkTransform->SetOffset(itkVec);

  return sp_itkTransform;
}

//-----------------------------------------------------------------------------
template <const unsigned int t_Dim>
static vtkSmartPointer<vtkMatrix4x4> _ConvertFromITKTransform(const itk::TransformBase &itkTransform) throw (itk::ExceptionObject)
{
  typedef itk::AffineTransform<double, t_Dim> __ITKTransformType;

  vtkSmartPointer<vtkMatrix4x4> sp_mat;

  sp_mat = vtkSmartPointer<vtkMatrix4x4>::New();
  sp_mat->Identity();

  if (std::string(itkTransform.GetNameOfClass()) == "AffineTransform" && itkTransform.GetOutputSpaceDimension() == t_Dim) 
  {
    const __ITKTransformType &itkAffineTransform = *static_cast<const __ITKTransformType*>(&itkTransform);
    const typename __ITKTransformType::MatrixType matrix = itkAffineTransform.GetMatrix();
    const typename __ITKTransformType::OutputVectorType trans = itkAffineTransform.GetOffset();

    unsigned int rInd, cInd;

    MITK_DEBUG << "Reading transform:\n";
    for (rInd = 0; rInd < t_Dim; rInd++) 
    {
      for (cInd = 0; cInd < t_Dim; cInd++) 
      {
        MITK_DEBUG << matrix(rInd,cInd) << " ";
        sp_mat->Element[rInd][cInd] = matrix(rInd,cInd);
      }

      sp_mat->Element[rInd][3] = trans[rInd];
      MITK_DEBUG << trans[rInd] << std::endl;
    }
  } 
  else 
  { 
    itkGenericExceptionMacro(<< "Failed to cast input transform to ITK affine transform.\nInput transform has type " << itkTransform.GetNameOfClass() << " (" << itkTransform.GetOutputSpaceDimension() << "D)\n");
  }

  return sp_mat;
}


} // End of nonamespace

//-----------------------------------------------------------------------------
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
  }
}

//-----------------------------------------------------------------------------
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

//-----------------------------------------------------------------------------
/** \brief Slot for saving transform to disk. */
void AffineTransformer::OnSaveTransform(std::string fileName)
{
  assert(m_CurrentDataNode);
  
  itk::TransformFileWriter::Pointer sp_writer;
  sp_writer = itk::TransformFileWriter::New();
  sp_writer->SetFileName(fileName.c_str());

  try 
  {

    vtkSmartPointer<vtkMatrix4x4> transform = mitk::AffineTransformDataNodeProperty::LoadTransformFromNode(DISPLAYED_TRANSFORM_KEY.c_str(), *(m_CurrentDataNode.GetPointer()));
    sp_writer->SetInput(_ConvertToITKTransform<3, false>(*transform));
    sp_writer->Update();

    MITK_DEBUG << "Writing of current transform to file: success";
  } 
  catch (itk::ExceptionObject &r_ex) 
  {
    MITK_ERROR << "Caught ITK exception:\n" << r_ex.what() << endl;
  }
}

//-----------------------------------------------------------------------------
/** \brief Slot for loading transform from disk. */
void AffineTransformer::OnLoadTransform(std::string fileName)
{
  assert(m_CurrentDataNode);
  itk::TransformFileReader::Pointer sp_transformIO;

  try 
  {
    sp_transformIO = itk::TransformFileReader::New();
    sp_transformIO->SetFileName(fileName.c_str());
    sp_transformIO->Update();

    if (sp_transformIO->GetTransformList()->size() == 0) 
    {
      MITK_ERROR << "ITK didn't find any transforms in " << fileName << endl;
      return;
    }

    vtkSmartPointer<vtkMatrix4x4> transformFromFile = _ConvertFromITKTransform<3> (*sp_transformIO->GetTransformList()->front());
    MITK_DEBUG << "Reading of transform from file: success";

    this->ApplyLoadedTransformToNode(transformFromFile, m_CurrentDataNode);

    mitk::DataStorage::SetOfObjects::ConstPointer children = this->GetDataStorage()->GetDerivations(m_CurrentDataNode);
    
    for (unsigned int i = 0; i < children->Size(); i++) 
    {
      this->ApplyLoadedTransformToNode(transformFromFile, children->GetElement(i));
    }

    OnResetTransformPushed();

    MITK_DEBUG << "Applied transform from file: success";

  } 
  catch (itk::ExceptionObject &r_itkEx) 
  {
    MITK_ERROR << "Transform " << fileName << " is incompatible with image.\n" << "Caught ITK exception:\n" << r_itkEx.what() << std::endl;
  }
}

//-----------------------------------------------------------------------------
/** \brief Slot for loading transform from disk. */
void AffineTransformer::OnApplyTransformPushed()
{
}

//-----------------------------------------------------------------------------
/** \brief Slot for resampling the current image. */
void AffineTransformer::OnResampleTransformPushed()
{
}

//-----------------------------------------------------------------------------
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

//-----------------------------------------------------------------------------
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

//-----------------------------------------------------------------------------
/** Called by _UpdateTransformationGeometry to set new transformations in the right properties of the node. */
void AffineTransformer::UpdateNodeProperties(const vtkSmartPointer<vtkMatrix4x4> displayedTransformFromParameters,
                          const vtkSmartPointer<vtkMatrix4x4> incrementalTransformToBeComposed,
                          mitk::DataNode::Pointer)
{
}

//-----------------------------------------------------------------------------
/** Called by _UpdateNodeProperties to update a transform property on a given node. */
void AffineTransformer::UpdateTransformProperty(std::string name, vtkSmartPointer<vtkMatrix4x4> transform, mitk::DataNode& node)
{
}

//-----------------------------------------------------------------------------
/** The transform loaded from file is applied to the current node, and all its children, and it resets the GUI parameters to Identity, and hence the DISPLAY_TRANSFORM and DISPLAY_PARAMETERS to Identity.*/
void AffineTransformer::ApplyLoadedTransformToNode(const vtkSmartPointer<vtkMatrix4x4> transformFromFile, mitk::DataNode::Pointer node)
{
  /**************************************************************
  * This is the main method to apply a transformation from file.
  **************************************************************/

  vtkSmartPointer<vtkMatrix4x4> incrementalTransformation = mitk::AffineTransformDataNodeProperty::LoadTransformFromNode(INCREMENTAL_TRANSFORM_KEY.c_str(), *(node.GetPointer()));
  vtkSmartPointer<vtkMatrix4x4> invertedIncrementalTransformation = vtkMatrix4x4::New();
  vtkMatrix4x4::Invert(incrementalTransformation, invertedIncrementalTransformation);
  node->GetData()->GetGeometry()->Compose( invertedIncrementalTransformation );

  mitk::AffineTransformDataNodeProperty::Pointer transformFromFileProperty = mitk::AffineTransformDataNodeProperty::New();
  transformFromFileProperty->SetTransform(*(transformFromFile.GetPointer()));
  node->ReplaceProperty(PRELOADED_TRANSFORM_KEY.c_str(), transformFromFileProperty);
  node->GetData()->GetGeometry()->Compose( transformFromFile );

  mitk::AffineTransformDataNodeProperty::Pointer affineTransformIdentity = mitk::AffineTransformDataNodeProperty::New();
  affineTransformIdentity->Identity();
  node->ReplaceProperty(DISPLAYED_TRANSFORM_KEY.c_str(), affineTransformIdentity);

  mitk::AffineTransformParametersDataNodeProperty::Pointer affineTransformParametersIdentity = mitk::AffineTransformParametersDataNodeProperty::New();
  affineTransformParametersIdentity->Identity();
  node->ReplaceProperty(DISPLAYED_PARAMETERS_KEY.c_str(), affineTransformParametersIdentity);
}

//-----------------------------------------------------------------------------
/** \brief Applies a re-sampling to the current node. */
void AffineTransformer::ApplyResampleToCurrentNode()
{
  assert(m_CurrentDataNode.IsNotNull());

  mitk::Image::Pointer image = dynamic_cast<mitk::Image*>(m_CurrentDataNode->GetData());
  assert(image);

  vtkSmartPointer<vtkMatrix4x4> sp_transformFromParams = this->ComputeTransformFromParameters();
  vtkSmartPointer<vtkMatrix4x4> sp_transformPreLoaded = mitk::AffineTransformDataNodeProperty::LoadTransformFromNode(PRELOADED_TRANSFORM_KEY.c_str(), *(m_CurrentDataNode.GetPointer()));
  vtkSmartPointer<vtkMatrix4x4> sp_combinedTransform = vtkMatrix4x4::New();
  vtkMatrix4x4::Multiply4x4(sp_transformFromParams, sp_transformPreLoaded, sp_combinedTransform);
  vtkSmartPointer<vtkMatrix4x4> sp_incTransform = vtkMatrix4x4::New();
  vtkMatrix4x4::Invert(sp_combinedTransform, sp_incTransform);

  try 
  {
    #define APPLY_MULTICHANNEL(TMultiChannelType) AccessFixedPixelTypeByItk_n(image, _ApplyTransformMultiChannel, ( TMultiChannelType ), (*sp_combinedTransform))

    if (image->GetPixelType().GetNumberOfComponents() == 3) 
    {
      #define APPLY_MULTICHANNEL_RGB(TBaseType) APPLY_MULTICHANNEL(itk::RGBPixel< TBaseType >)

      if (image->GetPixelType().GetTypeId() == typeid(signed char) && image->GetPixelType().GetBitsPerComponent() == 8) 
      {
        MITK_DEBUG << "Assuming RGB (signed char)\n" <<"ITK typeID: " << typeid(itk::RGBPixel<signed char>).name() << std::endl;

        APPLY_MULTICHANNEL_RGB(signed char);
      } 
      else if (image->GetPixelType().GetTypeId() == typeid(unsigned char) && image->GetPixelType().GetBitsPerComponent() == 8) 
      {
        MITK_DEBUG << "Assuming RGB (unsigned char)\n"
          << "ITK typeID: " << typeid(itk::RGBPixel<unsigned char>).name() << std::endl;

        APPLY_MULTICHANNEL_RGB(unsigned char);
      } 
      else if (image->GetPixelType().GetTypeId() == typeid(signed short) && image->GetPixelType().GetBitsPerComponent() == 16) 
      {
        APPLY_MULTICHANNEL_RGB(signed short);
      } 
      else if (image->GetPixelType().GetTypeId() == typeid(unsigned short) && image->GetPixelType().GetBitsPerComponent() == 16) 
      {
        APPLY_MULTICHANNEL_RGB(unsigned short);
      } 
      else if (image->GetPixelType().GetTypeId() == typeid(float) && image->GetPixelType().GetBitsPerComponent() == 32) 
      {
        MITK_DEBUG << "Assuming RGB (float)\n" << "ITK typeID: " << typeid(itk::RGBPixel<float>).name() << std::endl;

        APPLY_MULTICHANNEL_RGB(float);
      } 
      else if (image->GetPixelType().GetTypeId()== typeid(double) && image->GetPixelType().GetBitsPerComponent() == 64) 
      {
        APPLY_MULTICHANNEL_RGB(double);
      } 
      else 
      {
        MITK_ERROR << "pixel type " << image->GetPixelType().GetItkTypeAsString() << " is not supported.\n";
      }

      #undef APPLY_MULTICHANNEL_RGB
    } 
    else if (image->GetPixelType().GetNumberOfComponents() == 4) 
    {
      #define APPLY_MULTICHANNEL_RGBA(TBaseType) APPLY_MULTICHANNEL(itk::RGBAPixel< TBaseType >)

      if (image->GetPixelType().GetTypeId() == typeid(unsigned char) && image->GetPixelType().GetBitsPerComponent() == 8) 
      {
        MITK_DEBUG << "Assuming RGB (unsigned char)\n" << "ITK typeID: " << typeid(itk::RGBAPixel<unsigned char>).name() << std::endl;

        APPLY_MULTICHANNEL_RGBA(signed char);
      } 
      else if (image->GetPixelType().GetTypeId() == typeid(signed char) && image->GetPixelType().GetBitsPerComponent() == 8) 
      {
        MITK_DEBUG << "Assuming RGB (signed char)\n"
          << "ITK typeID: " << typeid(itk::RGBAPixel<signed char>).name() << std::endl;

        APPLY_MULTICHANNEL_RGBA(unsigned char);
      } 
      else if (image->GetPixelType().GetTypeId() == typeid(signed short) && image->GetPixelType().GetBitsPerComponent() == 16) 
      {
        APPLY_MULTICHANNEL_RGBA(signed short);
      } 
      else if (image->GetPixelType().GetTypeId() == typeid(unsigned short) && image->GetPixelType().GetBitsPerComponent() == 16) 
      {
        APPLY_MULTICHANNEL_RGBA(unsigned short);
      } 
      else if (image->GetPixelType().GetTypeId() == typeid(float) && image->GetPixelType().GetBitsPerComponent() == 32) 
      {
        MITK_DEBUG << "Assuming RGB (float)\n" << "ITK typeID: " << typeid(itk::RGBAPixel<float>).name() << std::endl;

        APPLY_MULTICHANNEL_RGBA(float);
      } 
      else if (image->GetPixelType().GetTypeId()== typeid(double) && image->GetPixelType().GetBitsPerComponent() == 64) 
      {
        APPLY_MULTICHANNEL_RGBA(double);
      } 
      else
        MITK_ERROR << "pixel type " << image->GetPixelType().GetItkTypeAsString() << " is not supported.\n";

      #undef APPLY_MULTICHANNEL_RGBA
    } 
    else 
    {
      AccessByItk_n(image, _ApplyTransform, (*sp_combinedTransform));
    }

    #undef APPLY_MULTICHANNEL
  } 
  catch (mitk::AccessByItkException &r_ex) 
  {
    MITK_ERROR << "MITK Exception:\n" << r_ex.what() << endl;
  }

  image->Modified();
  m_CurrentDataNode->Modified();
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


} // end namespace
