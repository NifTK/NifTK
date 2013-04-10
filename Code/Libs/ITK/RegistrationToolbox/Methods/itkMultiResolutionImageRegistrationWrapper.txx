/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef _itkMultiResolutionImageRegistrationWrapper_txx
#define _itkMultiResolutionImageRegistrationWrapper_txx

#include "itkMultiResolutionImageRegistrationWrapper.h"
#include "itkRecursiveMultiResolutionPyramidImageFilter.h"
#include "itkBinaryCrossStructuringElement.h"

#include "itkLogHelper.h"

namespace itk
{
/*
 * Constructor
 */
template < typename TInputImageType, class TPyramidFilter>
MultiResolutionImageRegistrationWrapper<TInputImageType, TPyramidFilter>
::MultiResolutionImageRegistrationWrapper()
{

  m_FixedImage   = 0; // has to be provided by the user.
  m_FixedMask    = 0; // has to be provided by the user.
  m_MovingImage  = 0; // has to be provided by the user.
  m_MovingMask   = 0; // has to be provided by the user.

  m_InitialTransformParametersOfNextLevel = ParametersType(1);
  m_InitialTransformParametersOfNextLevel.Fill( 0.0f );

  m_FixedImagePyramid  = ImagePyramidType::New();
  m_FixedMaskPyramid   = ImagePyramidType::New();  
  m_MovingImagePyramid = ImagePyramidType::New();
  m_MovingMaskPyramid  = ImagePyramidType::New();

  m_FixedMaskThresholder = ThresholdFilterType::New();
  m_MovingMaskThresholder = ThresholdFilterType::New();
  
  m_MaskBeforePyramid = false;
  
  m_UserSpecifiedSchedule = false;
  
  m_Stop = false;
  m_NumberOfLevels = 1;
  m_CurrentLevel = 0;
  m_StartLevel = std::numeric_limits<unsigned int>::max();
  m_StopLevel = std::numeric_limits<unsigned int>::max();
  m_UseOriginalImageAtFinalLevel = true; 
  
  niftkitkDebugMacro(<<"Constructed:MultiResolutionImageRegistrationWrapper, m_Stop=" << m_Stop \
      << ", m_NumberOfLevels=" << m_NumberOfLevels \
      << ", m_CurrentLevel=" << m_CurrentLevel \
      << ", m_StartLevel=" << m_StartLevel \
      << ", m_StopLevel=" << m_StopLevel \
      << ", m_UserSpecifiedSchedule=" << m_UserSpecifiedSchedule \
      << ", m_MaskBeforePyramid=" << m_MaskBeforePyramid
      );
}

/*
 * Stop the Registration Process
 */
template < typename TInputImageType, class TPyramidFilter>
void
MultiResolutionImageRegistrationWrapper<TInputImageType, TPyramidFilter>
::StopRegistration( void )
{
  niftkitkDebugMacro(<<"StopRegistration()");
  m_Stop = true;
}

/*
 * Stop the Registration Process
 */
template < typename TInputImageType, class TPyramidFilter>
void
MultiResolutionImageRegistrationWrapper<TInputImageType, TPyramidFilter>
::PreparePyramids( void )
{
  typedef ImageRegionIterator<TInputImageType> IteratorType;
  
  niftkitkDebugMacro(<<"PreparePyramids():Started, doing " << m_NumberOfLevels << " levels");
  
  // Sanity checks
  if( !m_FixedImage )
    {
      itkExceptionMacro(<<"PreparePyramids():FixedImage is not present");
    }

  if( !m_MovingImage )
    {
      itkExceptionMacro(<<"PreparePyramids():MovingImage is not present");
    }

  m_FixedImagePyramid->SetNumberOfLevels( m_NumberOfLevels );

  if (this->m_UserSpecifiedSchedule)
    {
      m_FixedImagePyramid->SetUseShrinkImageFilter(false); 
      m_FixedImagePyramid->SetSchedule(*m_Schedule);	
    }
  
  m_FixedImagePyramid->SetInput( m_FixedImage );  
  m_FixedImagePyramid->UpdateLargestPossibleRegion();
  
  m_MovingImagePyramid->SetNumberOfLevels( m_NumberOfLevels );

  if (this->m_UserSpecifiedSchedule)
    {
      m_MovingImagePyramid->SetUseShrinkImageFilter(false); 
      m_MovingImagePyramid->SetSchedule(*m_Schedule);	
    }
    
  m_MovingImagePyramid->SetInput( m_MovingImage );
  m_MovingImagePyramid->UpdateLargestPossibleRegion();

  if (!m_FixedMask.IsNull())
    {
      niftkitkDebugMacro(<<"PreparePyramids():Doing fixed mask");
      if (m_MaskBeforePyramid) 
        {
          // This has the curious side effect, that if your datatype is not float,
          // then when you set it to [0,1], when you put it into the pyramid, 
          // the smoothing and sub-sampling will produce non-integer values between 0 
          // and 1, which will be cast to zero. Hence most of the mask will disappear.
          niftkitkInfoMacro(<<"PreparePyramids():Binarising fixed mask before pyramid. Warning");
          m_FixedMaskThresholder->SetInput(m_FixedMask);
          m_FixedMaskThresholder->SetLowerThreshold(1);
          m_FixedMaskThresholder->SetUpperThreshold(std::numeric_limits<InputImagePixelType>::max());
          m_FixedMaskThresholder->SetInsideValue(1);
          m_FixedMaskThresholder->SetOutsideValue(0);
          m_FixedMaskThresholder->UpdateLargestPossibleRegion();
          m_FixedMaskPyramid->SetInput(m_FixedMaskThresholder->GetOutput());
          niftkitkDebugMacro(<<"PreparePyramids():Done Binarising fixed mask before pyramid");
        }
      else
        {
          m_FixedMaskPyramid->SetInput(m_FixedMask);    
        }
      m_FixedMaskPyramid->SetNumberOfLevels( m_NumberOfLevels );    
      if (this->m_UserSpecifiedSchedule)
        {
          m_FixedMaskPyramid->SetUseShrinkImageFilter(false); 
          m_FixedMaskPyramid->SetSchedule(*m_Schedule); 
        }
      m_FixedMaskPyramid->UpdateLargestPossibleRegion();
      niftkitkDebugMacro(<<"PreparePyramids():Done fixed mask");
    }

  if (!m_MovingMask.IsNull())
    {
      niftkitkDebugMacro(<<"PreparePyramids():Doing moving mask");
      if (m_MaskBeforePyramid) 
        {
          // This has the curious side effect, that if your datatype is not float,
          // then when you set it to [0,1], when you put it into the pyramid, 
          // the smoothing and sub-sampling will produce non-integer values between 0 
          // and 1, which will be cast to zero. Hence most of the mask will disappear.
          niftkitkInfoMacro(<<"PreparePyramids():Binarising moving mask before pyramid. Warning");
          m_MovingMaskThresholder->SetInput(m_MovingMask);
          m_MovingMaskThresholder->SetLowerThreshold(1);
          m_MovingMaskThresholder->SetUpperThreshold(std::numeric_limits<InputImagePixelType>::max());
          m_MovingMaskThresholder->SetInsideValue(1);
          m_MovingMaskThresholder->SetOutsideValue(0);
          m_MovingMaskThresholder->UpdateLargestPossibleRegion();
          m_MovingMaskPyramid->SetInput(m_MovingMaskThresholder->GetOutput());
          niftkitkDebugMacro(<<"PreparePyramids():Done Binarising moving mask before pyramid");
        }
      else
        {
          m_MovingMaskPyramid->SetInput(m_MovingMask);    
        }
      m_MovingMaskPyramid->SetNumberOfLevels( m_NumberOfLevels );    
      if (this->m_UserSpecifiedSchedule)
        {
          m_MovingMaskPyramid->SetUseShrinkImageFilter(false); 
          m_MovingMaskPyramid->SetSchedule(*m_Schedule); 
        }
      m_MovingMaskPyramid->UpdateLargestPossibleRegion();
      niftkitkDebugMacro(<<"PreparePyramids():Done fixed mask");
    }
    
  niftkitkDebugMacro(<<"PreparePyramids():Finished, doing " << m_NumberOfLevels << " levels");
}


/*
 * Initialize by setting the interconnects between components. 
 */
template < typename TInputImageType, class TPyramidFilter>
void
MultiResolutionImageRegistrationWrapper<TInputImageType, TPyramidFilter>
::Initialize() throw (ExceptionObject)
{
  // Sanity checks
  if ( !m_SingleResMethod )
    {
      itkExceptionMacro(<<"Initialize():Single resolution method not provided, so I can't register." );
    }

  InputImageConstPointer inputFixedImage;
  InputImageConstPointer inputMovingImage;
  InputImageConstPointer inputFixedMask;
  InputImageConstPointer inputMovingMask;
  
  if (m_UseOriginalImageAtFinalLevel && (m_NumberOfLevels == 1 || m_CurrentLevel == m_NumberOfLevels-1))
    {
      niftkitkDebugMacro(<<"Initialize():Not using pyramids");
      inputFixedImage = this->GetFixedImage();
      inputMovingImage = this->GetMovingImage();
      inputFixedMask = this->GetFixedMask();
      inputMovingMask = this->GetMovingMask();
    }
  else
    {
      niftkitkDebugMacro(<<"Initialize():Multi-res, m_CurrentLevel=" << m_CurrentLevel \
        << ", fixedImageSize=" << m_FixedImagePyramid->GetOutput(m_CurrentLevel)->GetLargestPossibleRegion().GetSize() \
        << ", fixedImageSpacing=" << m_FixedImagePyramid->GetOutput(m_CurrentLevel)->GetSpacing() \
        << ", fixedImageOrigin=" << m_FixedImagePyramid->GetOutput(m_CurrentLevel)->GetOrigin() \
        << ", fixedImageDirection=\n" << m_FixedImagePyramid->GetOutput(m_CurrentLevel)->GetDirection() \
        << ", movingImageSize=" << m_MovingImagePyramid->GetOutput(m_CurrentLevel)->GetLargestPossibleRegion().GetSize()  \
        << ", movingImageSpacing=" << m_MovingImagePyramid->GetOutput(m_CurrentLevel)->GetSpacing() \
        << ", movingImageOrigin=" << m_MovingImagePyramid->GetOutput(m_CurrentLevel)->GetOrigin() \
        << ", movingImageDirection=\n" << m_MovingImagePyramid->GetOutput(m_CurrentLevel)->GetDirection()
      );

      inputFixedImage = m_FixedImagePyramid->GetOutput(m_CurrentLevel);
      m_FixedImagePyramid->UpdateLargestPossibleRegion();
      
      inputMovingImage = m_MovingImagePyramid->GetOutput(m_CurrentLevel);
      m_MovingImagePyramid->UpdateLargestPossibleRegion();

      if (!m_FixedMask.IsNull())
        {
          inputFixedMask = m_FixedMaskPyramid->GetOutput(m_CurrentLevel);
          m_FixedMaskPyramid->UpdateLargestPossibleRegion();          
        }

      if (!m_MovingMask.IsNull())
        {
          inputMovingMask = m_MovingMaskPyramid->GetOutput(m_CurrentLevel);
          m_MovingMaskPyramid->UpdateLargestPossibleRegion();          
        }
    }

      // Calculate min and max image values in fixed image.
      ImageRegionConstIterator<TInputImageType> fiIt(inputFixedImage,
                                                  inputFixedImage->
                                                      GetLargestPossibleRegion());
      fiIt.GoToBegin();
      InputImagePixelType minFixed = fiIt.Value();
      InputImagePixelType maxFixed = fiIt.Value();
      ++fiIt;
      while ( !fiIt.IsAtEnd() )
        {
          InputImagePixelType value = fiIt.Value();

          if (value < minFixed)
            {
              minFixed = value;
            }
              else if (value > maxFixed)
            {
              maxFixed = value;
            }
          ++fiIt;
        }
      
      // Calculate min and max image values in moving image.
      ImageRegionConstIterator<TInputImageType> miIt(inputMovingImage,
                                                   inputMovingImage->
                                                       GetLargestPossibleRegion());
      miIt.GoToBegin();
      InputImagePixelType minMoving = miIt.Value();
      InputImagePixelType maxMoving = miIt.Value();
      ++miIt;
      while ( !miIt.IsAtEnd() )
        {
          InputImagePixelType value = miIt.Value();

          if (value < minMoving)
            {
              minMoving = value;
            }
          else if (value > maxMoving)
            {
              maxMoving = value;
            }
          ++miIt;
        }
      niftkitkDebugMacro(<<std::string("Initialize():Checking output of pyramids:")
        + "fixedLower:" + niftk::ConvertToString((double)minFixed)
        + ",fixedUpper:" + niftk::ConvertToString((double)maxFixed)
        + ",movingLower:" + niftk::ConvertToString((double)minMoving)
        + ",movingUpper:" + niftk::ConvertToString((double)maxMoving));

  niftkitkDebugMacro(<<"Initialize():Connecting up single-res registration object");
  
  m_SingleResMethod->SetFixedImage(inputFixedImage);
  m_SingleResMethod->SetMovingImage(inputMovingImage);
  
  if (!m_FixedMask.IsNull())
    {
      niftkitkDebugMacro(<<"Initialize():Setting fixed mask onto singleResMethod");
      m_SingleResMethod->SetUseFixedMask(true); 
      m_SingleResMethod->SetFixedMask(inputFixedMask);
    }
  else
    {
      m_SingleResMethod->SetUseFixedMask(false);
      niftkitkDebugMacro(<<"Initialize():Not using fixed mask");
    }
  
  if (!m_MovingMask.IsNull())
    {
      niftkitkDebugMacro(<<"Initialize():Setting moving mask onto singleResMethod");
      m_SingleResMethod->SetUseMovingMask(true); 
      m_SingleResMethod->SetMovingMask(inputMovingMask);
    }
  else
    {
      m_SingleResMethod->SetUseMovingMask(false); 
      niftkitkDebugMacro(<<"Initialize():Not using moving mask");
    }
    
  niftkitkDebugMacro(<<"Initialize multi-res... DONE");
}


/*
 * Starts the Registration Process
 */
template < typename TInputImageType, class TPyramidFilter>
void
MultiResolutionImageRegistrationWrapper<TInputImageType, TPyramidFilter>
::StartRegistration( void )
{ 
  niftkitkDebugMacro(<<"StartRegistration():Starting");
  
  m_Stop = false;

  if (m_StopLevel == std::numeric_limits<unsigned int>::max())
    {
      m_StopLevel = m_NumberOfLevels - 1;
      niftkitkDebugMacro(<<"StartRegistration():Stop level wasn't set, so defaulting to:" <<  m_StopLevel);
    }
  
  if (m_StartLevel == std::numeric_limits<unsigned int>::max())
    {
      m_StartLevel = 0;
      niftkitkDebugMacro(<<"StartRegistration():Start level wasn't set, so defaulting to:" <<  m_StartLevel);
    }

  this->PreparePyramids();

  for ( m_CurrentLevel = 0; m_CurrentLevel < m_NumberOfLevels ; m_CurrentLevel++ )
    {
      this->InvokeEvent( IterationEvent() );
    
      // Check if there has been a stop request
      if ( m_Stop ) 
        {
          niftkitkDebugMacro(<<"Stop requested");
          break;
        }

      // This connects the right image from the pyramid.
      this->Initialize();

      // Any other preparation, implemented in subclasses.
      this->BeforeSingleResolutionRegistration();

      // make sure we carry transformation parameters between levels!!!
      m_SingleResMethod->SetInitialTransformParameters(m_InitialTransformParametersOfNextLevel);
      
      // We have a mechanism to only do certain levels.
      niftkitkInfoMacro(<<"StartRegistration():Starting level:" << m_CurrentLevel
          << ", with " << m_InitialTransformParametersOfNextLevel.GetSize() 
          << " parameters, m_StartLevel=" << m_StartLevel 
          << ", m_StopLevel=" << m_StopLevel);

      if (m_CurrentLevel >= m_StartLevel && m_CurrentLevel <= m_StopLevel)            
        {
          this->m_SingleResMethod->Update();    
        }
      
      // Any other post-processing, implemented in subclasses.
      this->AfterSingleResolutionRegistration();

      // setup the initial parameters for next level
      if ( m_CurrentLevel < m_NumberOfLevels - 1 )
        {
          m_InitialTransformParametersOfNextLevel = m_SingleResMethod->GetLastTransformParameters();
        }
        
    } // end for each level
    
  niftkitkDebugMacro(<<"StartRegistration():Finished");
      
} // end function


/*
 * PrintSelf
 */
template < typename TInputImageType, class TPyramidFilter>
void
MultiResolutionImageRegistrationWrapper<TInputImageType, TPyramidFilter>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf( os, indent );

  os << indent << "FixedImage: " << m_FixedImage.GetPointer() << std::endl;
  os << indent << "MovingImage: " << m_MovingImage.GetPointer() << std::endl;
  os << indent << "FixedImagePyramid: " << m_FixedImagePyramid.GetPointer() << std::endl;
  os << indent << "MovingImagePyramid: " << m_MovingImagePyramid.GetPointer() << std::endl;
  os << indent << "NumberOfLevels: " << m_NumberOfLevels << std::endl;
  os << indent << "CurrentLevel: " << m_CurrentLevel << std::endl;  
  os << indent << "InitialTransformParametersOfNextLevel: " << m_InitialTransformParametersOfNextLevel << std::endl;

  if (!m_SingleResMethod.IsNull())
    {
      os << indent << "m_SingleResMethod:" << std::endl;
      m_SingleResMethod.GetPointer()->Print(os, indent.GetNextIndent());
    }
  else
    {
      os << indent << "m_SingleResMethod: NULL" << std::endl;
    }
}


/*
 * Generate Data
 */
template < typename TInputImageType, class TPyramidFilter>
void
MultiResolutionImageRegistrationWrapper<TInputImageType, TPyramidFilter>
::GenerateData()
{
  niftkitkDebugMacro(<<"GenerateData():Starting");
  
  this->StartRegistration();
  
  niftkitkDebugMacro(<<"GenerateData():Finished");
}



template < typename TInputImageType, class TPyramidFilter>
unsigned long
MultiResolutionImageRegistrationWrapper<TInputImageType, TPyramidFilter>
::GetMTime() const
{
  unsigned long mtime = Superclass::GetMTime();
  unsigned long m;


  // Some of the following should be removed once ivars are put in the
  // input and output lists
  
  if (m_SingleResMethod)
    {
    m = m_SingleResMethod->GetMTime();
    mtime = (m > mtime ? m : mtime);
    }

  if (m_FixedImage)
    {
    m = m_FixedImage->GetMTime();
    mtime = (m > mtime ? m : mtime);
    }

  if (m_MovingImage)
    {
    m = m_MovingImage->GetMTime();
    mtime = (m > mtime ? m : mtime);
    }

  return mtime;
  
}

/*
 *  Get Output
 */
template < typename TInputImageType, class TPyramidFilter>
const typename MultiResolutionImageRegistrationWrapper<TInputImageType, TPyramidFilter>::TransformOutputType *
MultiResolutionImageRegistrationWrapper<TInputImageType, TPyramidFilter>
::GetOutput() const
{
  return static_cast< const TransformOutputType * >( this->ProcessObject::GetOutput(0) );
}



template < typename TInputImageType, class TPyramidFilter>
DataObject::Pointer
MultiResolutionImageRegistrationWrapper<TInputImageType, TPyramidFilter>
::MakeOutput(unsigned int output)
{
  switch (output)
    {
    case 0:
      return static_cast<DataObject*>(TransformOutputType::New().GetPointer());
      break;
    default:
      itkExceptionMacro("MakeOutput request for an output number larger than the expected number of outputs");
      return 0;
    }
}

} // end namespace itk


#endif
