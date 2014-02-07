/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef Undistortion_h
#define Undistortion_h

#include "niftkOpenCVExports.h"
#include <mitkDataStorage.h>
#include <mitkCameraIntrinsics.h>
#include <opencv2/core/types_c.h>
#include <mitkImage.h>


namespace niftk
{

/**
 * Idea is to create one of these Undistortion objects for a given DataNode or Image,
 * which then does all the magic for computing the undistortion and caches some data.
 * Output, however, is not tied into an instance of the class. Instead it's passed into Run().
 *
 * How to use.
 * You can pass in any combination of input DataNode or Image and output DataNode or Image.
 * 1) For exmaple:
 *    mitk::DataNode::Pointer   input = ...;
 *    mitk::DataNode::Pointer   output = mitk::DataNode::New();
 *    niftk::Undistortion*      undist = new niftk::Undistortion(input);
 *    undist->Run(output);
 *
 * 2) Another one:
 *    mitk::Image::Pointer   input = ...;
 *    mitk::Image::Pointer   output;
 *    niftk::Undistortion*   undist = new niftk::Undistortion(input);
 *    undist->PrepareOutput(output);
 *    undist->Run(output);
 *
 * Either way, the node or image need to have calibration properties. The output object will
 * have these calibrations copied to it as well, including s_ImageIsUndistortedPropertyName.
 * In case both node and its attached image have calibration props then image wins. Also,
 * for consistency sake, both calibration props need to match in this case.
 *
 * A note about smart pointers: various functions take as parameter something like
 * const mitk::DataNode::Pointer&. This means that the smart-pointer itself is const but the
 * pointed-to object is not! Const is not transitive for smart-pointers!
 * Using const smart-pointers avoids unnecessary ref-count operations (which make stepping through
 * the code rather annoying). If you wanted a const pointed-to object then the parameter would
 * be mitk::DataNode::ConstPointer.
 *
 * When I wrote this originally, I decided for a non-mitk-smartypants class here, i.e. bare-bones
 * c++ class. Now I can't remember why. Feel free to change it to mitk.
 *
 * @warning Only 2D images are tested and supported! Anything else might or might not work, crash,
 * trigger the apocolypse. For 3D what probably happens is that only the very first slice is processed.
 */
class NIFTKOPENCV_EXPORT Undistortion
{
public:
  static const char* s_ImageIsUndistortedPropertyName;       // mitk::BoolProperty

  // FIXME: this one should go to our calibration class/module, not here
  static const char* s_CameraCalibrationPropertyName; // mitk::CameraIntrinsicsProperty
  // FIXME: i dont think this is the best place to keep these. i'm up for suggestions!
  static const char* s_ImageIsRectifiedPropertyName;         // mitk::BoolProperty
  static const char* s_StereoRigTransformationPropertyName;  // niftk::MatrixProperty

  // used for stereo-rig transformation, i.e. between left and right camera
 typedef mitk::GenericProperty<itk::Matrix<float, 4, 4> > MatrixProperty;

public:
  /**
   * node should have Image data attached, at least when Run() is called.
   * @throws std::runtime_error if node.IsNull()
   */
  Undistortion(const mitk::DataNode::Pointer& node);

  /**
   * @throws std::runtime_error if image.IsNull()
   */
  Undistortion(const mitk::Image::Pointer& image);

  virtual ~Undistortion();

public:
  // loads calibration from a text file (not the opencv xml format!).
  // if filename is empty then it will dream up some parameters for the given image.
  static void LoadIntrinsicCalibration(const std::string& filename, mitk::DataNode::Pointer node);
  static void LoadIntrinsicCalibration(const std::string& filename, const mitk::Image::Pointer& img);
  static void LoadStereoRig(const std::string& filename, mitk::DataNode::Pointer node);
  static void LoadStereoRig(const std::string& filename, mitk::Image::Pointer img);
  static bool NeedsToLoadIntrinsicCalib(const std::string& filename, const mitk::DataNode::Pointer& node);
  static bool NeedsToLoadIntrinsicCalib(const std::string& filename, const mitk::Image::Pointer& image);
  static bool NeedsToLoadStereoRigExtrinsics(const std::string& filename, const mitk::DataNode::Pointer& node);
  static bool NeedsToLoadStereoRigExtrinsics(const std::string& filename, const mitk::Image::Pointer& image);

  /**
   * \brief Copies properties from node to image, if they don't already exist on the image.
   */
  static void CopyImagePropsIfNecessary(const mitk::DataNode::Pointer source, mitk::Image::Pointer target);


  /**
   * @warning You need to call output->Modified() yourself if you want listeners to be notified!
   * @throws stuff from ValidateInput().
   * @throws std::runtime_error if output is null.
   * @throws std::runtime_error if the attached image is zero-size.
   * @post If successful, output node will have an image attached of the correct size.
   */
  virtual void Run(const mitk::DataNode::Pointer& output);

  /**
   * output has to have the same size as input (which you passed in to the constructor).
   * Call PrepareOutput() to make sure.
   */
  virtual void Run(const mitk::Image::Pointer& output);

  /**
   * You should call this if you use Run(const mitk::Image::Pointer&).
   * If output is null then a new image is allocated.
   * If the dimensions of an existing image are wrong then a new one is allocated too.
   *
   * @throw std::runtime_error rethrown from ValidateInput()
   */
  void PrepareOutput(mitk::Image::Pointer& output);

protected:
  /**
   * Check that we have an image to work on, it has the correct depth/channels, etc.
   * @throws std::runtime_error if image type is wrong or required properties are missing.
   * @throws std::runtime_error if image is zero-size.
   */
  void ValidateInput();

  /**
   * FIXME: presumably this is virtual so that we could derive a gpu version.
   *        but then the ipl parameters are no use!
   * @param recomputeCache true image dimensions or type have changed, m_RecomputeCache is passed in (and set to false afterwards, by Run()).
   * @throws exceptions if anything is wrong.
   */
  virtual void Process(const IplImage* input, IplImage* output, bool recomputeCache);

protected:

  static bool NeedsToLoadImageProperty(const std::string& fileName,
                                       const std::string& propertyName,
                                       const mitk::Image::Pointer& image);

  // the node that this class is to operate on
  mitk::DataNode::Pointer     m_Node;
  // the image attached to our node
  mitk::Image::Pointer        m_Image;
  // the intrinsic parameters belonging to the image
  mitk::CameraIntrinsics::Pointer     m_Intrinsics;

  // cached for repeated use in Process()
  IplImage*        m_MapX;
  IplImage*        m_MapY;
  bool             m_RecomputeCache;
};


} // namespace


#endif // niftk_Undistortion_h
