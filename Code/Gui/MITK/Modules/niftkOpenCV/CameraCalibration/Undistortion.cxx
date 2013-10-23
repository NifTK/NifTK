/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "Undistortion.h"
#include <mitkCameraIntrinsicsProperty.h>
#include <mitkImageReadAccessor.h>
#include <mitkImageWriteAccessor.h>
#include <mitkProperties.h>
#include <Conversion/ImageConversion.h>


namespace niftk
{

const char* Undistortion::s_CameraCalibrationPropertyName       = "niftk.CameraCalibration";
const char* Undistortion::s_ImageIsUndistortedPropertyName      = "niftk.ImageIsUndistorted";
const char* Undistortion::s_ImageIsRectifiedPropertyName        = "niftk.ImageIsRectified";
const char* Undistortion::s_StereoRigTransformationPropertyName = "niftk.StereoRigTransformation";

//-----------------------------------------------------------------------------
Undistortion::Undistortion(mitk::DataNode::Pointer node)
  : m_Node(node), m_MapX(0), m_MapY(0)
{
}


//-----------------------------------------------------------------------------
Undistortion::Undistortion(mitk::Image::Pointer image)
  : m_Image(image), m_MapX(0), m_MapY(0)
{
}


//-----------------------------------------------------------------------------
Undistortion::~Undistortion()
{
  if (m_MapX)
  {
    cvReleaseImage(&m_MapX);
  }
  if (m_MapY)
  {
    cvReleaseImage(&m_MapY);
  }
}


//-----------------------------------------------------------------------------
void Undistortion::LoadIntrinsicCalibration(const std::string& filename, mitk::DataNode::Pointer node)
{
  mitk::Image::Pointer  img = dynamic_cast<mitk::Image*>(node->GetData());
  LoadIntrinsicCalibration(filename, img);
  node->SetProperty(s_CameraCalibrationPropertyName, img->GetProperty(s_CameraCalibrationPropertyName));
}


//-----------------------------------------------------------------------------
void Undistortion::LoadIntrinsicCalibration(const std::string& filename, mitk::Image::Pointer img)
{
  assert(img.IsNotNull());

  mitk::CameraIntrinsics::Pointer    cam = mitk::CameraIntrinsics::New();

  if (!filename.empty())
  {
    // FIXME: we need to try different formats: plain text, opencv's xml
    std::ifstream   file(filename.c_str());
    if (!file.good())
    {
      throw std::runtime_error("Cannot open calibration file " + filename);
    }
    float   values[9 + 4];
    for (unsigned int i = 0; i < (sizeof(values) / sizeof(values[0])); ++i)
    {
      if (!file.good())
      {
        throw std::runtime_error("Cannot read enough data from calibration file " + filename);
      }
      file >> values[i];
    }
    file.close();

    cam->SetFocalLength(values[0], values[4]);
    cam->SetPrincipalPoint(values[2], values[5]);
    cam->SetDistorsionCoeffs(values[9], values[10], values[11], values[12]);
  }
  else
  {
    // invent some stuff based on image dimensions
    unsigned int w = img->GetDimension(0);
    unsigned int h = img->GetDimension(1);

    mitk::Point3D::ValueType  focal[3] = {(float) std::max(w, h), (float) std::max(w, h), 1};
    mitk::Point3D::ValueType  princ[3] = {(float) w / 2, (float) h / 2, 1};
    mitk::Point4D::ValueType  disto[4] = {0, 0, 0, 0};

    cam->SetIntrinsics(mitk::Point3D(focal), mitk::Point3D(princ), mitk::Point4D(disto));
  }

  mitk::CameraIntrinsicsProperty::Pointer   prop = mitk::CameraIntrinsicsProperty::New(cam);
  img->SetProperty(s_CameraCalibrationPropertyName, prop);
}


//-----------------------------------------------------------------------------
void Undistortion::LoadStereoRig(const std::string& filename,
                                 mitk::DataNode::Pointer node)
{
  mitk::Image::Pointer  img = dynamic_cast<mitk::Image*>(node->GetData());
  LoadStereoRig(filename, img);
  node->SetProperty(s_StereoRigTransformationPropertyName, img->GetProperty(s_StereoRigTransformationPropertyName));
}


//-----------------------------------------------------------------------------
void Undistortion::LoadStereoRig(
    const std::string& filename,
    mitk::Image::Pointer img)
{
  assert(img.IsNotNull());

  itk::Matrix<float, 4, 4>    txf;
  txf.SetIdentity();

  if (!filename.empty())
  {
    std::ifstream   file(filename.c_str());
    if (!file.good())
    {
      throw std::runtime_error("Cannot open stereo-rig file " + filename);
    }
    float   values[3 * 4];
    for (unsigned int i = 0; i < (sizeof(values) / sizeof(values[0])); ++i)
    {
      if (!file.good())
      {
        throw std::runtime_error("Cannot read enough data from stereo-rig file " + filename);
      }
      file >> values[i];
    }
    file.close();

    // set rotation
    for (int i = 0; i < 9; ++i)
    {
      txf.GetVnlMatrix()(i / 3, i % 3) = values[i];
    }

    // set translation
    for (int i = 0; i < 3; ++i)
    {
      txf.GetVnlMatrix()(i, 3) = values[9 + i];
    }
  }
  else
  {
    // no idea what to invent here...
  }

  // Attach as property to image.
  MatrixProperty::Pointer  prop = MatrixProperty::New(txf);
  img->SetProperty(s_StereoRigTransformationPropertyName, prop);
}


//-----------------------------------------------------------------------------
bool Undistortion::NeedsToLoadImageProperty(const std::string& fileName,
                                            const std::string& propertyName,
                                            const mitk::Image::Pointer& image)
{
  bool  needs2load = false;

  // filename overrides any existing properties
  if (fileName.size() > 0)
  {
    needs2load = true;
  }
  else
  {
    mitk::BaseProperty::Pointer prop = image->GetProperty(propertyName.c_str());
    if (prop.IsNull())
    {
      needs2load = true;
    }
  }
  return needs2load;
}


//-----------------------------------------------------------------------------
bool Undistortion::NeedsToLoadIntrinsicCalib(const std::string& fileName, const mitk::DataNode::Pointer& node)
{
  mitk::Image::Pointer  img = dynamic_cast<mitk::Image*>(node->GetData());
  return NeedsToLoadIntrinsicCalib(fileName, img);
}


//-----------------------------------------------------------------------------
bool Undistortion::NeedsToLoadIntrinsicCalib(const std::string& fileName, const mitk::Image::Pointer& image)
{
  return NeedsToLoadImageProperty(fileName, s_CameraCalibrationPropertyName, image);
}


//-----------------------------------------------------------------------------
bool Undistortion::NeedsToLoadStereoRigExtrinsics(const std::string& fileName, const mitk::DataNode::Pointer& node)
{
  mitk::Image::Pointer  img = dynamic_cast<mitk::Image*>(node->GetData());
  return NeedsToLoadStereoRigExtrinsics(fileName, img);
}


//-----------------------------------------------------------------------------
bool Undistortion::NeedsToLoadStereoRigExtrinsics(const std::string& fileName, const mitk::Image::Pointer& image)
{
  return NeedsToLoadImageProperty(fileName, s_StereoRigTransformationPropertyName, image);
}



//-----------------------------------------------------------------------------
template <typename PropType>
static void CopyProp(const mitk::DataNode::Pointer source, mitk::Image::Pointer target, const char* name)
{
  mitk::BaseProperty::Pointer baseProp = target->GetProperty(name);
  if (baseProp.IsNull())
  {
    // none there yet, try to pull it from the datanode
    baseProp = source->GetProperty(name);
    if (baseProp.IsNotNull())
    {
      // check that it's the correct type
      typename PropType::Pointer   prop = dynamic_cast<PropType*>(baseProp.GetPointer());
      if (prop.IsNotNull())
      {
        // FIXME: copy? or simply ref the same object?
        target->SetProperty(name, prop);
      }
    }
  }
}


//-----------------------------------------------------------------------------
void Undistortion::CopyImagePropsIfNecessary(const mitk::DataNode::Pointer source, mitk::Image::Pointer target)
{
  // we copy known meta-data properties to the images, but only if they dont exist yet.
  // we'll not ever change the value of an existing property!

  // calibration data
  CopyProp<mitk::CameraIntrinsicsProperty>(source, target, niftk::Undistortion::s_CameraCalibrationPropertyName);

  // has the image been rectified?
  CopyProp<mitk::BoolProperty>(source, target, niftk::Undistortion::s_ImageIsRectifiedPropertyName);

  // has the image been undistorted?
  CopyProp<mitk::BoolProperty>(source, target, niftk::Undistortion::s_ImageIsUndistortedPropertyName);
}


//-----------------------------------------------------------------------------
void Undistortion::Process(const IplImage* input, IplImage* output, bool recomputeCache)
{
  if (recomputeCache)
  {
    if (m_MapX)
    {
      cvReleaseImage(&m_MapX);
    }
    if (m_MapY)
    {
      cvReleaseImage(&m_MapY);
    }
  }

  assert(m_Intrinsics.IsNotNull());

  if (m_MapX == 0)
  {
    assert(m_MapY == 0);
  
    m_MapX = cvCreateImage(cvSize(input->width, input->height), IPL_DEPTH_32F, 1);
    m_MapY = cvCreateImage(cvSize(input->width, input->height), IPL_DEPTH_32F, 1);

    // the old-style CvMat will reference the memory in the new-style cv::Mat.
    // that's why we keep these in separate variables.
    cv::Mat   cammat  = m_Intrinsics->GetCameraMatrix();
    cv::Mat   distmat = m_Intrinsics->GetDistorsionCoeffs();
    CvMat cam  = cammat;
    CvMat dist = distmat;
    cvInitUndistortMap(&cam, &dist, m_MapX, m_MapY);
  }

  cvRemap(input, output, m_MapX, m_MapY, CV_INTER_LINEAR /*+CV_WARP_FILL_OUTLIERS*/, cvScalarAll(0));
}


//-----------------------------------------------------------------------------
void Undistortion::ValidateInput(bool& recomputeCache)
{
  mitk::CameraIntrinsics::Pointer   nodeIntrinsic;

  // do we have a node?
  // if so try to find the attached image
  // if not check if we have been created with an image
  if (m_Node.IsNotNull())
  {
    // note that the image data attached to the node overrides whatever previous image we had our hands on
    mitk::BaseData::Pointer data = m_Node->GetData();
    if (data.IsNotNull())
    {
      mitk::Image::Pointer img = dynamic_cast<mitk::Image*>(data.GetPointer());
      if (img.IsNotNull())
      {
        m_Image = img;
      }
      else
      {
        throw std::runtime_error("DataNode has non-image data attached");
      }
    }
    else
    {
      throw std::runtime_error("DataNode does not have any data object attached");
    }

    // getting here... we have a DataNode with an Image attached
    mitk::BaseProperty::Pointer cambp = m_Node->GetProperty(s_CameraCalibrationPropertyName);
    if (cambp.IsNotNull())
    {
      mitk::CameraIntrinsicsProperty::Pointer cam = dynamic_cast<mitk::CameraIntrinsicsProperty*>(cambp.GetPointer());
      // node has a property of the right name but the type is wrong
      if (cam.IsNull())
      {
        throw std::runtime_error("Wrong property type for camera calibration data");
      }
      nodeIntrinsic = cam->GetValue();
    }
  }

  // we have either been created with a node or with an image
  if (m_Image.IsNull())
  {
    throw std::runtime_error("No image data to work on");
  }

  int numComponents     = m_Image->GetPixelType().GetNumberOfComponents();
  int bitsPerComponent  = m_Image->GetPixelType().GetBitsPerComponent();
  // very limited set for now
  if (numComponents != 3 && numComponents != 4)
  {
    throw std::runtime_error("Only 3 or 4 component images supported");
  }
  if (bitsPerComponent != 8)
  {
    throw std::runtime_error("Only images with 8 bit depth supported");
  }

  // we may or may not have checked yet whether image has a property
  // even if node has one already we check anyway for sanity-check purposes
  mitk::BaseProperty::Pointer   imgprop = m_Image->GetProperty(s_CameraCalibrationPropertyName);
  if (imgprop.IsNotNull())
  {
    mitk::CameraIntrinsicsProperty::Pointer imgcalib = dynamic_cast<mitk::CameraIntrinsicsProperty*>(imgprop.GetPointer());
    if (imgcalib.IsNotNull())
    {
      // if both node and attached image have calibration props then both need to match!
      // otherwise something in our system causes inconsistent data.
      mitk::CameraIntrinsics::Pointer c = imgcalib->GetValue();
      if (nodeIntrinsic.IsNotNull())
      {
        if (!c->Equals(nodeIntrinsic.GetPointer()))
        {
          throw std::runtime_error("Inconsistent calibration data on DataNode and its Image");
        }
      }
      else
      {
        nodeIntrinsic = c;
      }
    }
    else
    {
      throw std::runtime_error("Wrong property type for camera calibration data");
    }
  }


  if (nodeIntrinsic.IsNull())
  {
    throw std::runtime_error("No camera calibration data found");
  }

  if (m_Intrinsics.IsNotNull())
  {
    recomputeCache = !m_Intrinsics->Equals(nodeIntrinsic.GetPointer());
    if (recomputeCache)
    {
      // we need to copy it because somebody could modify that instance of CameraIntrinsics
      m_Intrinsics = nodeIntrinsic->Clone();
    }
  }
  else
  {
    // we need to copy it because somebody could modify that instance of CameraIntrinsics
    m_Intrinsics = nodeIntrinsic->Clone();
  }

  // input image size can change wrt our previously cached remap image.
  if (m_MapX != 0)
  {
    // these always come in pairs.
    assert(m_MapY != 0);

    if ((m_Image->GetDimension(0) != (unsigned int) m_MapX->width) ||
        (m_Image->GetDimension(1) != (unsigned int) m_MapX->height))
    {
      recomputeCache = true;
    }
  }
  // FIXME: check that calibration data is in some meaningful range for our input image
}


//-----------------------------------------------------------------------------
void Undistortion::PrepareOutput(mitk::Image::Pointer& outputImage)
{
  if (!outputImage.IsNull())
  {
    bool haswrongsize = false;
    haswrongsize |= outputImage->GetDimension(0) != m_Image->GetDimension(0);
    haswrongsize |= outputImage->GetDimension(1) != m_Image->GetDimension(1);
    haswrongsize |= outputImage->GetDimension(2) != 1;

    if (haswrongsize)
    {
      outputImage = mitk::Image::Pointer();
    }
  }


  if (outputImage.IsNull())
  {
    // this is pretty disgusting stuff
    IplImage* temp = cvCreateImage(cvSize(m_Image->GetDimension(0), m_Image->GetDimension(1)), m_Image->GetPixelType().GetBitsPerComponent(), m_Image->GetPixelType().GetNumberOfComponents());
    outputImage = CreateMitkImage(temp);
    cvReleaseImage(&temp);
  }

  mitk::Geometry3D::Pointer   geomp = m_Image->GetGeometry();
  // FIXME: should clone it!
  //mitk::Geometry3D* geom = geomp->Clone().GetPointer();
  outputImage->SetGeometry(geomp);
}


//-----------------------------------------------------------------------------
void Undistortion::Run(mitk::DataNode::Pointer output)
{
  mitk::Image::Pointer outputImage = dynamic_cast<mitk::Image*>(output->GetData());
  PrepareOutput(outputImage);
  output->SetData(outputImage);

  Run(outputImage);

  output->SetProperty(s_CameraCalibrationPropertyName, mitk::CameraIntrinsicsProperty::New(m_Intrinsics));
  output->SetProperty(s_ImageIsUndistortedPropertyName, mitk::BoolProperty::New(true));
  // this will kick off itk listeners. not a good idea in multi-threaded situations.
  //output->Modified();
}


//-----------------------------------------------------------------------------
void Undistortion::Run(mitk::Image::Pointer outputImage)
{
  bool  recomputeCache = true;
  ValidateInput(recomputeCache);

  // always lock output first! so that we are not blocking previous stages if
  // our subsequent one is slow.
  mitk::ImageWriteAccessor  outputAccess(outputImage);
  void* outputPointer = outputAccess.GetData();
  mitk::ImageReadAccessor   inputAccess(m_Image);
  const void* inputPointer = inputAccess.GetData();

  IplImage  outipl;
  cvInitImageHeader(&outipl, cvSize((int) outputImage->GetDimension(0), (int) outputImage->GetDimension(1)), outputImage->GetPixelType().GetBitsPerComponent(), outputImage->GetPixelType().GetNumberOfComponents());
  cvSetData(&outipl, outputPointer, outputImage->GetDimension(0) * (outputImage->GetPixelType().GetBitsPerComponent() / 8) * outputImage->GetPixelType().GetNumberOfComponents());

  IplImage  inipl;
  cvInitImageHeader(&inipl, cvSize((int) m_Image->GetDimension(0), (int) m_Image->GetDimension(1)), m_Image->GetPixelType().GetBitsPerComponent(), m_Image->GetPixelType().GetNumberOfComponents());
  cvSetData(&inipl, const_cast<void*>(inputPointer), m_Image->GetDimension(0) * (m_Image->GetPixelType().GetBitsPerComponent() / 8) * m_Image->GetPixelType().GetNumberOfComponents());

  Process(&inipl, &outipl, recomputeCache);

  // copy relevant props to output image
  outputImage->SetProperty(s_CameraCalibrationPropertyName, mitk::CameraIntrinsicsProperty::New(m_Intrinsics));
  outputImage->SetProperty(s_ImageIsUndistortedPropertyName, mitk::BoolProperty::New(true));
  // this will kick off itk listeners. not a good idea in multi-threaded situations.
  //outputImage->Modified();
}


} // namespace
