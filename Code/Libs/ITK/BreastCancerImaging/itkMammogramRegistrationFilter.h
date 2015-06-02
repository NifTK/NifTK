/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkMammogramRegistrationFilter_h
#define itkMammogramRegistrationFilter_h

#include <itkImageToImageFilter.h>

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

#include <itkImageMomentsCalculator.h>
#include <itkImageRegistrationFactory.h>
#include <itkMammogramMaskSegmentationImageFilter.h>

/** \class MammogramRegistrationFilter
 * \brief Registers a pair of 2D mammograms.
 */


namespace itk
{


template< class TInputImage, class TOutputImage >
class MammogramRegistrationFilter :
class MammogramRegistrationFilter :
  public ImageToImageFilter< TInputImage, TOutputImage >
{
public:

  /// Standard class typedefs.
  typedef MammogramRegistrationFilter Self;
  typedef ImageToImageFilter< TInputImage, TOutputImage > Superclass;
  typedef SmartPointer< Self > Pointer;

  /// Method for creation through the object factory.
  itkNewMacro(Self);

  /// Run-time type information (and related methods).
  itkTypeMacro(MammogramRegistrationFilter, ImageToImageFilter);

  /// Image dimension.
  itkStaticConstMacro(ImageDimension, unsigned int,
                      TInputImage::ImageDimension);

  /// Type of the input image
  typedef TInputImage                           InputImageType;
  typedef typename InputImageType::Pointer      InputImagePointer;
  typedef typename InputImageType::ConstPointer InputImageConstPointer;
  typedef typename InputImageType::RegionType   InputImageRegionType;
  typedef typename InputImageType::PixelType    InputImagePixelType;
  typedef typename InputImageType::SpacingType  InputImageSpacingType;
  typedef typename InputImageType::PointType    InputImagePointType;

  typedef typename NumericTraits<InputImagePixelType>::RealType    RealType;

  /// Type of the output image
  typedef TOutputImage                          OutputImageType;
  typedef typename OutputImageType::Pointer     OutputImagePointer;
  typedef typename OutputImageType::RegionType  OutputImageRegionType;
  typedef typename OutputImageType::PixelType   OutputImagePixelType;
  typedef typename OutputImageType::IndexType   OutputImageIndexType;
  typedef typename OutputImageType::PointType   OutputImagePointType;


  typedef typename itk::Vector< float, ImageDimension > VectorPixelType;
  typedef typename itk::Image< VectorPixelType, ImageDimension > DeformationFieldType;

  typedef typename itk::ImageFileReader< InputImageType > FileReaderType;
  typedef typename itk::ImageFileWriter< InputImageType > FileWriterType;



  typedef double ScalarType;

  typedef typename itk::ImageRegistrationFactory< InputImageType, ImageDimension, ScalarType > FactoryType;

  typedef typename FactoryType::EulerAffineTransformType EulerAffineTransformType;
  typedef typename EulerAffineTransformType::Pointer EulerAffinePointerType;

  typedef typename itk::ImageMomentsCalculator< InputImageType > ImageMomentCalculatorType;

  typedef typename itk::MammogramMaskSegmentationImageFilter<InputImageType, InputImageType> MammogramMaskFilterType;


  void SetVerboseOn( void ) { m_FlgVerbose = true; }
  void SetVerboseOff( void ) { m_FlgVerbose = false; }

  void SetDebugOn( void ) { m_FlgDebug = true; }
  void SetDebugOff( void ) { m_FlgDebug = false; }

  void SetOverwriteRegistration( void ) { m_FlgOverwrite = true; }

  /// Set the directory to look for executables in
  void SetExecutablesDirectoryToTry( std::string dir ) { m_DirExecutable = dir; }

  /// The input images to register
  typedef enum {
    REGISTER_ORIGINAL_IMAGES,
    REGISTER_MASKS,
    REGISTER_DISTANCE_TRANSFORMS
  } enumRegistrationImagesType;

  /// Set the registration image type.
  void SetTypeOfInputImagesToRegister( enumRegistrationImagesType regImagesType ) {
    m_TypeOfInputImagesToRegister = regImagesType;
  }

  /// Set the final control point spacing for non-rigid registrations

  itkSetMacro( ControlPointSpacing, float );
  /// Get the final control point spacing for non-rigid registrations
  itkGetMacro( ControlPointSpacing, float );

  /// Set the number of multi-scale registration levels
  itkSetMacro( NumberOfLevels, unsigned int );
  /// Get the number of multi-scale registration levels
  itkGetMacro( NumberOfLevels, unsigned int );

  /// Set the number of multi-scale registration levels to use
  itkSetMacro( NumberOfLevelsToUse, unsigned int );
  /// Get the number of multi-scale registration levels to use
  itkGetMacro( NumberOfLevelsToUse, unsigned int );


  /// Set the target image.
  void SetTargetImage(InputImagePointer imTarget) { this->SetInput(0, imTarget); }

  /// Set the source image.
  void SetSourceImage(InputImagePointer imSource) { this->SetInput(1, imSource); }

  /// Set the target mask.
  void SetTargetMask(InputImagePointer maskTarget) { this->SetInput(2, maskTarget); }

  /// Set the source mask.
  void SetSourceMask(InputImagePointer maskSource) { this->SetInput(3, maskSource); }

  /// Set the target registration mask.
  void SetTargetRegnMask(InputImagePointer maskTargetRegn) { this->SetInput(4, maskTargetRegn); }



  /// Set the working directory for storing any intermediate files
  void SetWorkingDirectory( std::string directory ) {
    m_DirWorking = directory;
  }


  /// Set the target image filename
  void SetFileTarget( std::string filename ) {
    m_FileTarget = filename;
  }
  /// Set the source image filename
  void SetFileSource( std::string filename ) {
    m_FileSource = filename;
  }

  /// Set the target mask image filename
  void SetFileTargetMask( std::string filename ) {
    m_FileTargetMask = filename;
  }
  /// Set the source mask image filename
  void SetFileSourceMask( std::string filename ) {
    m_FileSourceMask = filename;
  }

  /// Set the input target registration mask filename
  void SetFileInputTargetRegistrationMask( std::string filename ) {
    m_FileInputTargetRegistrationMask = filename;
  }

  /// Set the output target registration mask filename
  void SetFileOutputTargetRegistrationMask( std::string fileOut ) {
    m_FileOutputTargetRegistrationMask = fileOut;
  }

  /// Set the output target mask distance transform image
  void SetFileTargetDistanceTransform( std::string fileOut ) {
    m_FileTargetDistanceTransform = fileOut;
  }
  /// Set the output source mask distance transform image
  void SetFileSourceDistanceTransform( std::string fileOut ) {
    m_FileSourceDistanceTransform = fileOut;
  }

  /// Set the output affine transformation matrix file
  void SetFileOutputAffineTransformation( std::string fileOut ) {
    m_FileOutputAffineTransformation = fileOut;
  }
  /// Set the output non-rigid transformation (control-point) file
  void SetFileOutputNonRigidTransformation( std::string fileOut ) {
    m_FileOutputNonRigidTransformation = fileOut;
  }
  /// Set the output deformation field file
  void SetFileOutputDeformation( std::string fileOut ) {
    m_FileOutputDeformation = fileOut;
  }

  /// Set the output affine registered file
  void SetFileOutputAffineRegistered( std::string fileOut ) {
    m_FileOutputAffineRegistered = fileOut;
  }
  /// Set the output non-rigid registered file
  void SetFileOutputNonRigidRegistered( std::string fileOut ) {
    m_FileOutputNonRigidRegistered = fileOut;
  }

  /// Specify whether to perform a non-rigid registration
  void SetRegisterNonRigid( void ) { m_FlgRegisterNonRigid = true; }

  /// Transform a point from the target image to the source
  InputImagePointType TransformPoint( InputImagePointType point );

  /// Non-rigidly trasnform an image file using the registration
  void NonRigidlyTransformImageFile( std::string fileImage, std::string fileResult );

#ifdef ITK_USE_CONCEPT_CHECKING
  /// Begin concept checking
  itkConceptMacro(DimensionShouldBe2,
                  (Concept::SameDimension<itkGetStaticConstMacro(InputImageDimension),2>));
  itkConceptMacro(InputHasNumericTraitsCheck,
                  (Concept::HasNumericTraits<InputImagePixelType>));
  itkConceptMacro(OutputHasPixelTraitsCheck,
                  (Concept::HasPixelTraits<OutputImagePixelType>));
  /// End concept checking
#endif


protected:
  MammogramRegistrationFilter();
  ~MammogramRegistrationFilter(){}

  bool m_FlgVerbose;
  bool m_FlgDebug;

  /// Overwrite existing registration even if it already exists
  bool m_FlgOverwrite;

  /// Specify whether to perform a non-rigid registration
  bool m_FlgRegisterNonRigid;

  /// Specify the input images to register
  enumRegistrationImagesType m_TypeOfInputImagesToRegister;

  /// The final control point spacing in mm
  float m_ControlPointSpacing;

  /// The number of multi-scale registration levels
  unsigned int m_NumberOfLevels;
  /// The number of multi-scale registration levels to use
  unsigned int m_NumberOfLevelsToUse;

  /// A working directory for storing any intermediate files
  std::string m_DirWorking;
  /// A directory to look for executables in
  std::string m_DirExecutable;

  /// The input target image filename
  std::string m_FileTarget;
  /// The input source image filename
  std::string m_FileSource;

  /// The input target mask image filename
  std::string m_FileTargetMask;
  /// The input source mask image filename
  std::string m_FileSourceMask;

  /// The input target registration mask filename
  std::string m_FileInputTargetRegistrationMask;

  /// The output target registration mask filename
  std::string m_FileOutputTargetRegistrationMask;

  /// The output target mask distance transform image
  std::string m_FileTargetDistanceTransform;
  /// The output source mask distance transform image
  std::string m_FileSourceDistanceTransform;

  /// The output affine transformation matrix
  std::string m_FileOutputAffineTransformation;
  /// The output non-rigid transformation
  std::string m_FileOutputNonRigidTransformation;

  /// The output deformation field
  std::string m_FileOutputDeformation;

  /// The output affine registered image
  std::string m_FileOutputAffineRegistered;
  /// The output non-rigidly registered image
  std::string m_FileOutputNonRigidRegistered;

  /// The target image
  InputImagePointer m_Target;
  /// The source image
  InputImagePointer m_Source;

  /// The target mask image
  InputImagePointer m_TargetMask;
  /// The source mask image
  InputImagePointer m_SourceMask;

  /// The target registration mask image
  InputImagePointer m_TargetRegnMask;

  /// The affine transformation
  EulerAffinePointerType m_AffineTransform;

  /// The registration deformation field
  typename DeformationFieldType::Pointer m_DeformationField;

  /// Check if an image is nifti and create a nifti version if not
  std::string ImageFileIsNiftiOrConvert( std::string fileInput );

  /// Compute the distance transform of a mask
  InputImagePointer GetDistanceTransform( InputImagePointer imMask );

  /// Expand the target mask to include the breast edge
  InputImagePointer GetTargetRegistrationMask( InputImagePointer imMask );

  /// Initialise the affine transform using the image moments
  void InitialiseTransformationFromImageMoments( InputImagePointer imTarget,
                                                 InputImagePointer imSource );

  /// Create a new filename for a subsampled image
  std::string AddSamplingFactorSuffix( std::string fileInput, float ampling );
  /// Create a subsampled image if any of the image dimensions exceed 2048
  std::string FileOfImageWithDimensionsLessThan2048( std::string fileInput );

  /// Print the object
  void Print( void );

  /// The affine registration
  InputImagePointer RunAffineRegistration( InputImagePointer imTarget,
                                           InputImagePointer imSource,
                                           int finalInterpolator=4,
                                           int registrationInterpolator=2 );
  /// The a non-rigid registration
  InputImagePointer RunNonRigidRegistration( void );

  /// Read the affine transformation
  bool ReadAffineTransformation( std::string fileAffineTransformation );
  /// Read the affine transformation
  bool ReadNonRigidDeformationField( std::string fileInputDeformation );

  /// Read the transformations instead of re-running them?
  bool ReadRegistrationData();

  /// Does the real work
  virtual void GenerateData();

  ///  Create the Output
  DataObject::Pointer MakeOutput(unsigned int idx);

private:
  MammogramRegistrationFilter(const Self &); //purposely not implemented
  void operator=(const Self &);  //purposely not implemented
};


} // namespace itk


#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMammogramRegistrationFilter.txx"
#endif

#endif


