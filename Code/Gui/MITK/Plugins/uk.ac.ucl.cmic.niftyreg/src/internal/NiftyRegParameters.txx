/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: $
 Revision          : $Revision: $
 Last modified by  : $Author:  $

 Original author   : j.hipwell@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include <ostream>

#include "mitkImageToNifti.h"
#include "niftiImageToMitk.h"



// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

template <class PRECISION_TYPE>
NiftyRegParameters<PRECISION_TYPE>::NiftyRegParameters()
{

  m_ReferenceImage = 0;
  m_FloatingImage = 0;
  m_ReferenceMaskImage = 0;
  m_ControlPointGridImage = 0;

  SetDefaultParameters();

}


// ---------------------------------------------------------------------------
// DeallocateImages();
// --------------------------------------------------------------------------- 

template <class PRECISION_TYPE>
void
NiftyRegParameters<PRECISION_TYPE>::DeallocateImages( void )
{
  if ( m_ReferenceImage ) 
  {
    nifti_image_free( m_ReferenceImage );
    m_ReferenceImage = 0;
  }

  if ( m_FloatingImage ) 
  {
    nifti_image_free( m_FloatingImage );
    m_FloatingImage = 0;
  }
    
  if ( m_ReferenceMaskImage )
  {
    nifti_image_free( m_ReferenceMaskImage );
    m_ReferenceMaskImage = 0;
  }
    
  if ( m_ControlPointGridImage )
  {
    nifti_image_free( m_ControlPointGridImage );
    m_ControlPointGridImage = 0;
  }
}


// ---------------------------------------------------------------------------
// SetDefaultParameters()
// --------------------------------------------------------------------------- 

template <class PRECISION_TYPE>
void NiftyRegParameters<PRECISION_TYPE>::SetDefaultParameters()
{

  // Multi-Scale Options
    
  m_LevelNumber = 3;		// Number of level to perform
  m_Level2Perform = 3;		// Only perform the first levels 

  // Input Image Options

  m_TargetSigmaValue = 0;  // Smooth the target image using the specified sigma (mm) 
  m_SourceSigmaValue = 0;  // Smooth the source image using the specified sigma (mm)

  // Flag indicating whether to do rigid and/or non-rigid registrations

  m_FlagDoInitialRigidReg = true;
  m_FlagDoNonRigidReg = true;


  // Initial affine transformation
 
  m_FlagInputAffine = false;
  m_FlagFlirtAffine = false;

  m_InputAffineName.clear();


  // The 'reg_aladin' parameters
  m_AladinParameters.SetDefaultParameters();

  // The 'reg_f3d' parameters
  m_F3dParameters.SetDefaultParameters();

}


// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------

template <class PRECISION_TYPE>
NiftyRegParameters<PRECISION_TYPE>::~NiftyRegParameters()
{
  DeallocateImages();
}



// ---------------------------------------------------------------------------
// PrintSelf
// ---------------------------------------------------------------------------

template <class PRECISION_TYPE>
void NiftyRegParameters<PRECISION_TYPE>::PrintSelf( std::ostream& os )
{

  os << "NiftyReg-Number of multi-resolution levels: " <<  m_LevelNumber << std::endl;
  os << "NiftyReg-Number of (coarse to fine) multi-resolution levels: " << m_Level2Perform << std::endl;    

  os << "NiftyReg-Target image smoothing sigma (mm): " << m_TargetSigmaValue << std::endl;
  os << "NiftyReg-Source image smoothing sigma (mm): " << m_SourceSigmaValue << std::endl;

  os << "NiftyReg-Initial rigid registration flag: " << m_FlagDoInitialRigidReg << std::endl;
  os << "NiftyReg-Non-rigid registration flag: " << m_FlagDoNonRigidReg << std::endl;


  // Initial affine transformation
 
  os << "# Initial affine transformation" << std::endl;

  if ( m_InputAffineName.isEmpty() )
    os << "NiftyReg-InputAffineName: UNSET" << std::endl;
  else
    os << "NiftyReg-InputAffineName: " << m_InputAffineName.toStdString() << std::endl;

  os << "NiftyReg-InputAffineFlag: " << m_FlagInputAffine << std::endl;
  os << "NiftyReg-FlirtAffineFlag: " << m_FlagFlirtAffine << std::endl;

  		               
  m_AladinParameters.PrintSelf( os );
  m_F3dParameters.PrintSelf( os );
}


// ---------------------------------------------------------------------------
// operator=
// ---------------------------------------------------------------------------

template <class PRECISION_TYPE>
NiftyRegParameters<PRECISION_TYPE> 
&NiftyRegParameters<PRECISION_TYPE>::operator=(const NiftyRegParameters<PRECISION_TYPE> &p)
{

  m_LevelNumber = p.m_LevelNumber;
  m_Level2Perform = p.m_Level2Perform;    

  m_TargetSigmaValue = p.m_TargetSigmaValue;
  m_SourceSigmaValue = p.m_SourceSigmaValue;

  m_FlagDoInitialRigidReg = p.m_FlagDoInitialRigidReg;
  m_FlagDoNonRigidReg = p.m_FlagDoNonRigidReg;

  m_InputAffineName = p.m_InputAffineName;

  m_FlagInputAffine = p.m_FlagInputAffine;
  m_FlagFlirtAffine = p.m_FlagFlirtAffine;


  m_AladinParameters = p.m_AladinParameters;
  m_F3dParameters = p.m_F3dParameters;

  return *this;
}


// ---------------------------------------------------------------------------
// CreateAladinRegistrationObject();
// --------------------------------------------------------------------------- 

template <class PRECISION_TYPE>
reg_aladin<PRECISION_TYPE> *NiftyRegParameters<PRECISION_TYPE>
::CreateAladinRegistrationObject( mitk::Image *mitkSourceImage, 
				  mitk::Image *mitkTargetImage, 
				  mitk::Image *mitkTargetMaskImage )
{
  reg_aladin<PRECISION_TYPE> *REG;

  if ( m_AladinParameters.symFlag )
  {
    REG = new reg_aladin_sym<PRECISION_TYPE>;

    if ( mitkTargetMaskImage )
      std::cerr << "[NiftyReg Warning] You have a target image mask specified." << std::endl
		<< "[NiftyReg Warning] As no source image mask is specified," << std::endl
		<< "[NiftyReg Warning] the degree of symmetry will be limited." << std::endl;
  }
  else
    REG = new reg_aladin<PRECISION_TYPE>;
  

  // Get nifti versions of the images

  if ( m_FloatingImage ) nifti_image_free( m_FloatingImage );
  m_FloatingImage  = ConvertMitkImageToNifti<PRECISION_TYPE>( mitkSourceImage );

  if ( m_ReferenceImage ) nifti_image_free( m_ReferenceImage );
  m_ReferenceImage = ConvertMitkImageToNifti<PRECISION_TYPE>( mitkTargetImage );

  // Check the dimensions of the images

  reg_checkAndCorrectDimension( m_ReferenceImage );
  reg_checkAndCorrectDimension( m_FloatingImage );

  // Set the reference and floating image

  REG->SetInputReference( m_ReferenceImage );
  REG->SetInputFloating( m_FloatingImage );

  // Set the reference mask image 

  if ( mitkTargetMaskImage ) 
  {
    if ( m_ReferenceMaskImage ) nifti_image_free( m_ReferenceMaskImage );
    m_ReferenceMaskImage = ConvertMitkImageToNifti<PRECISION_TYPE>( mitkTargetMaskImage );

    reg_checkAndCorrectDimension(m_ReferenceMaskImage);

    // check the dimensions

    for ( int i=1; i<=m_ReferenceImage->dim[0]; i++ ) {
    
      if ( m_ReferenceImage->dim[i] != m_ReferenceMaskImage->dim[i] ) 
      {
	fprintf(stderr,"* ERROR The reference image and its mask do not have the same dimension\n");
	return 0;
      }
    }
    
    REG->SetInputMask( m_ReferenceMaskImage );
  }

  // Aladin - Initialisation
  
  REG->SetNumberOfLevels( m_LevelNumber );
  REG->SetLevelsToPerform( m_Level2Perform );
  
  REG->SetReferenceSigma( m_TargetSigmaValue );
  REG->SetFloatingSigma( m_SourceSigmaValue );
  
  if ( m_FlagInputAffine 
       && ( ! m_InputAffineName.isEmpty() ) )
    
    REG->SetInputTransform( strdup( m_InputAffineName.toStdString().c_str() ), 
			    m_FlagFlirtAffine );
  
  REG->SetAlignCentre( m_AladinParameters.alignCenterFlag );

  // Aladin - Method

  REG->SetPerformAffine( ( m_AladinParameters.regnType == RIGID_THEN_AFFINE )
			 || ( m_AladinParameters.regnType == DIRECT_AFFINE ) );

  REG->SetPerformRigid( ( m_AladinParameters.regnType == RIGID_ONLY )
			|| ( m_AladinParameters.regnType == RIGID_THEN_AFFINE ) );

  REG->SetMaxIterations( m_AladinParameters.maxiterationNumber );

  REG->SetBlockPercentage( m_AladinParameters.block_percent_to_use );
  REG->SetInlierLts( m_AladinParameters.inlier_lts );

  // Aladin - Advanced

  REG->SetInterpolation( m_AladinParameters.interpolation );

  return REG;
}


// ---------------------------------------------------------------------------
// CreateNonRigidRegistrationObject();
// --------------------------------------------------------------------------- 

template <class PRECISION_TYPE>
reg_f3d<PRECISION_TYPE> *NiftyRegParameters<PRECISION_TYPE>
::CreateNonRigidRegistrationObject( mitk::Image *mitkSourceImage, 
				    mitk::Image *mitkTargetImage, 
				    mitk::Image *mitkTargetMaskImage )
{
  // Get nifti versions of the images

  if ( m_ReferenceImage ) nifti_image_free( m_ReferenceImage );
  m_ReferenceImage = ConvertMitkImageToNifti<PRECISION_TYPE>( mitkTargetImage );

  if ( m_FloatingImage ) nifti_image_free( m_FloatingImage );
  m_FloatingImage = ConvertMitkImageToNifti<PRECISION_TYPE>( mitkSourceImage );

#if 0
  nifti_set_filenames( m_ReferenceImage,"f3dReference.nii",0,0 );
  nifti_image_write( m_ReferenceImage );

  nifti_set_filenames( m_FloatingImage,"f3dFloating.nii",0,0 );
  nifti_image_write( m_FloatingImage );
#endif

  // Check the dimensions of the images

  reg_checkAndCorrectDimension( m_ReferenceImage );
  reg_checkAndCorrectDimension( m_FloatingImage );

  // Set the reference mask image 

  if ( mitkTargetMaskImage )
  {
    if ( m_ReferenceMaskImage ) nifti_image_free( m_ReferenceMaskImage );
    m_ReferenceMaskImage = ConvertMitkImageToNifti<PRECISION_TYPE>( mitkTargetMaskImage );

    reg_checkAndCorrectDimension( m_ReferenceMaskImage );

    // check the dimensions

    for ( int i=1; i<=m_ReferenceImage->dim[0]; i++ )
    {
    
      if ( m_ReferenceImage->dim[i] != m_ReferenceMaskImage->dim[i] ) 
      {
	fprintf(stderr,"* ERROR The reference image and its mask do not have the same dimension\n");
	return 0;
      }
    }
  }

  // Read the input control point grid image

  if ( ! m_F3dParameters.inputControlPointGridName.isEmpty() ) 
  {

    if ( m_ControlPointGridImage ) nifti_image_free( m_ControlPointGridImage );
    m_ControlPointGridImage = nifti_image_read( m_F3dParameters.inputControlPointGridName
					      .toStdString().c_str(), true );

    if ( m_ControlPointGridImage == NULL ) 
    {
      fprintf(stderr, 
	      "Error when reading the input control point grid image %s\n",
	      m_F3dParameters.inputControlPointGridName.toStdString().c_str());
      return 0;
    }
    
    reg_checkAndCorrectDimension( m_ControlPointGridImage );
  }

  // Read the affine transformation

  mat44 *affineTransformation = NULL;

  if ( ( ! m_FlagDoInitialRigidReg ) &&
       m_FlagInputAffine && 
       ( ! m_InputAffineName.isEmpty() ) ) 
  {
    
    affineTransformation = (mat44 *) malloc( sizeof( mat44 ) );
    
    // Check first if the specified affine file exist
    
    if ( FILE *aff = fopen( m_InputAffineName.toStdString().c_str(), "r") ) 
    {
      fclose( aff );
    }
    else 
    {
      fprintf( stderr, "The specified input affine file (%s) can not be read\n",
	       m_InputAffineName.toStdString().c_str() );
      return 0;
    }
    
    reg_tool_ReadAffineFile( affineTransformation,
			     m_ReferenceImage,
			     m_FloatingImage,
			     strdup( m_InputAffineName.toStdString().c_str() ),
			     m_FlagFlirtAffine );
  }

  // Create the reg_f3d object

  reg_f3d<PRECISION_TYPE> *REG = NULL;

#ifdef _USE_CUDA

  CUdevice dev;
  CUcontext ctx;

  if ( m_F3dParameters.useGPU 
       && ( ! ( m_F3dParameters.linearEnergyWeight0 ||
		m_F3dParameters.linearEnergyWeight1 ) ) )
  {

    if ( ( m_ReferenceImage->dim[4] == 1 && 
	   m_FloatingImage->dim[4]  == 1 ) || 
	 ( m_ReferenceImage->dim[4] == 2 &&
	   m_FloatingImage->dim[4]  == 2 ) ) {

      // The CUDA card is setup

      cuInit(0);

      struct cudaDeviceProp deviceProp;     
      int device_count = 0;      

      cudaGetDeviceCount( &device_count );

      int device = m_F3dParameters.cardNumber;
      
      if ( m_F3dParameters.cardNumber == -1 ) 
      {
	
	// following code is from cutGetMaxGflopsDeviceId()
	
	int max_gflops_device = 0;
	int max_gflops = 0;
	int current_device = 0;

	while ( current_device < device_count ) 
	{
	  cudaGetDeviceProperties( &deviceProp, current_device );

	  int gflops = deviceProp.multiProcessorCount * deviceProp.clockRate;

	  if ( gflops > max_gflops ) 
	  {
	    max_gflops = gflops;
	    max_gflops_device = current_device;
	  }
	  ++current_device;
	}
	device = max_gflops_device;
      }
      
      NR_CUDA_SAFE_CALL(cudaSetDevice( device ));
      NR_CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, device ));

      cuDeviceGet(&dev,device);
      cuCtxCreate(&ctx, 0, dev);

      if ( deviceProp.major < 1 ) 
      {
	printf("[NiftyReg ERROR CUDA] The specified graphical card does not exist.\n");
	return 0;
      }
       
      REG = new reg_f3d_gpu(m_ReferenceImage->nt, m_FloatingImage->nt);

    }
    else
    {
      fprintf(stderr,
	      "[NiftyReg ERROR] The GPU implementation only handles "
	      "1 to 1 or 2 to 2 image(s) registration\n");
      exit(1);
    }
  }
  
  else

#endif // _USE_CUDA
    
  {
    
    REG = new reg_f3d<PRECISION_TYPE>( m_ReferenceImage->nt, 
				      m_FloatingImage->nt );

  }

  // Set the reg_f3d parameters

  REG->SetReferenceImage( m_ReferenceImage );
  REG->SetFloatingImage( m_FloatingImage );

  REG->PrintOutInformation();

  if ( mitkTargetMaskImage )
    REG->SetReferenceMask( m_ReferenceMaskImage );

  if ( m_ControlPointGridImage != NULL )
    REG->SetControlPointGridImage( m_ControlPointGridImage );

  if ( affineTransformation != NULL )
    REG->SetAffineTransformation( affineTransformation );
  
  REG->SetBendingEnergyWeight( m_F3dParameters.bendingEnergyWeight );
    
  REG->SetLinearEnergyWeights( m_F3dParameters.linearEnergyWeight0,
			       m_F3dParameters.linearEnergyWeight1 );
  
  REG->SetJacobianLogWeight( m_F3dParameters.jacobianLogWeight );
  
  if ( m_F3dParameters.jacobianLogApproximation )
    REG->ApproximateJacobianLog();
  else 
    REG->DoNotApproximateJacobianLog();

  REG->ApproximateParzenWindow();

  REG->SetMaximalIterationNumber( m_F3dParameters.maxiterationNumber );

  REG->SetReferenceSmoothingSigma( m_TargetSigmaValue );
  REG->SetFloatingSmoothingSigma( m_SourceSigmaValue );

  // NB: -std::numeric_limits<PRECISION_TYPE>::max() is a special value which 
  // indicates the maximum value for ThresholdUp and the minimum for ThresholdLow.

  if ( m_F3dParameters.referenceThresholdUp == -std::numeric_limits<PRECISION_TYPE>::max() )
    REG->SetReferenceThresholdUp( 0, std::numeric_limits<PRECISION_TYPE>::max() );
  else
    REG->SetReferenceThresholdUp( 0, m_F3dParameters.referenceThresholdUp );

  REG->SetReferenceThresholdLow( 0, m_F3dParameters.referenceThresholdLow );

  if ( m_F3dParameters.floatingThresholdUp == -std::numeric_limits<PRECISION_TYPE>::max() )
    REG->SetFloatingThresholdUp( 0, std::numeric_limits<PRECISION_TYPE>::max() );
  else
    REG->SetFloatingThresholdUp( 0, m_F3dParameters.floatingThresholdUp );

  REG->SetFloatingThresholdLow( 0, m_F3dParameters.floatingThresholdLow );

  REG->SetReferenceBinNumber( 0, m_F3dParameters.referenceBinNumber );
  REG->SetFloatingBinNumber( 0, m_F3dParameters.floatingBinNumber );
  
  if ( m_F3dParameters.warpedPaddingValue == -std::numeric_limits<PRECISION_TYPE>::max() )
    REG->SetWarpedPaddingValue( std::numeric_limits<PRECISION_TYPE>::quiet_NaN() );
  else
    REG->SetWarpedPaddingValue( m_F3dParameters.warpedPaddingValue );

  for ( unsigned int s=0; s<3; s++ )
    REG->SetSpacing( s, m_F3dParameters.spacing[s] );

  REG->SetLevelNumber( m_LevelNumber );
  REG->SetLevelToPerform( m_Level2Perform );

  REG->SetGradientSmoothingSigma( m_F3dParameters.gradientSmoothingSigma );

  if ( m_F3dParameters.similarity == SSD_SIMILARITY )
    REG->UseSSD();
  else
    REG->DoNotUseSSD();

  if ( m_F3dParameters.similarity == KLDIV_SIMILARITY )
    REG->UseKLDivergence();
  else 
    REG->DoNotUseKLDivergence();

  if ( m_F3dParameters.useConjugate )
    REG->UseConjugateGradient();
  else 
    REG->DoNotUseConjugateGradient();

  if ( m_F3dParameters.noPyramid )
    REG->DoNotUsePyramidalApproach();

  if ( m_F3dParameters.interpolation == CUBIC_INTERPOLATION )
    REG->UseCubicSplineInterpolation();
  else if ( m_F3dParameters.interpolation == LINEAR_INTERPOLATION )
    REG->UseLinearInterpolation();
  else if ( m_F3dParameters.interpolation == NEAREST_INTERPOLATION )
    REG->UseNeareatNeighborInterpolation();


    // Run the registration
#ifdef _USE_CUDA
    if (m_F3dParameters.useGPU && m_F3dParameters.checkMem) {
        size_t free, total, requiredMemory = REG->CheckMemoryMB_f3d();
        cuMemGetInfo(&free, &total);
        printf("[NiftyReg CUDA] The required memory to run the registration is %lu Mb\n",
               (unsigned long int)requiredMemory);
        printf("[NiftyReg CUDA] The GPU card has %lu Mb from which %lu Mb are currenlty free\n",
               (unsigned long int)total/(1024*1024), (unsigned long int)free/(1024*1024));
    }
#endif

    return REG;
}


