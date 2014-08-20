/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkCreatePositiveMammogram_h
#define __itkCreatePositiveMammogram_h

#include <itksys/SystemTools.hxx>

#include <itkImage.h>
#include <itkMetaDataDictionary.h>
#include <itkMetaDataObject.h>

#include <itkInvertIntensityBetweenMaxAndMinImageFilter.h>
#include <itkMinimumMaximumImageCalculator.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkLogNonZeroIntensitiesImageFilter.h>

namespace itk
{



// -------------------------------------------------------------------------
// SetTag()
// -------------------------------------------------------------------------

void SetTag( itk::MetaDataDictionary &dictionary,
	     std::string tagID,
	     std::string newTagValue )
{
  typedef itk::MetaDataDictionary DictionaryType;
  typedef itk::MetaDataObject< std::string > MetaDataStringType;

  // Search for the tag
  
  DictionaryType::ConstIterator tagItr = dictionary.Find( tagID );
  DictionaryType::ConstIterator end = dictionary.End();
   
  if ( tagItr != end )
  {
    MetaDataStringType::ConstPointer entryvalue = 
      dynamic_cast<const MetaDataStringType *>( tagItr->second.GetPointer() );
    
    if ( entryvalue )
    {
      std::string tagValue = entryvalue->GetMetaDataObjectValue();
      
      std::cout << "Changing tag (" << tagID <<  ") "
		<< " from: " << tagValue 
		<< " to: " << newTagValue << std::endl;
      
      itk::EncapsulateMetaData<std::string>( dictionary, tagID, newTagValue );
    }
  }
  else
  {
    std::cout << "Setting tag (" << tagID <<  ") "
	      << " to: " << newTagValue << std::endl;
      
    itk::EncapsulateMetaData<std::string>( dictionary, tagID, newTagValue );
  }

};


// -------------------------------------------------------------------------
// ModifyTag()
// -------------------------------------------------------------------------

void ModifyTag( itk::MetaDataDictionary &dictionary,
                std::string tagID,
                std::string newTagValue )
{
  typedef itk::MetaDataDictionary DictionaryType;
  typedef itk::MetaDataObject< std::string > MetaDataStringType;

  // Search for the tag
  
  std::string tagValue;
  
  DictionaryType::ConstIterator tagItr = dictionary.Find( tagID );
  DictionaryType::ConstIterator tagEnd = dictionary.End();
   
  if ( tagItr != tagEnd )
  {
    MetaDataStringType::ConstPointer entryvalue = 
      dynamic_cast<const MetaDataStringType *>( tagItr->second.GetPointer() );
    
    if ( entryvalue )
    {
      std::string tagValue = entryvalue->GetMetaDataObjectValue();
      
      std::cout << "Modifying tag (" << tagID <<  ") "
		<< " from: " << tagValue 
		<< " to: " << newTagValue << std::endl;
      
      itk::EncapsulateMetaData<std::string>( dictionary, tagID, newTagValue );
    }
  }

};


//  --------------------------------------------------------------------------
/// Create a positive version of a DICOM mammogram
//  --------------------------------------------------------------------------

template < typename TImage >
void CreatePositiveMammogram( typename TImage::Pointer &image,
                              itk::MetaDataDictionary &dictionary,
                              bool flgInvert=false )
{
  typedef itk::MetaDataDictionary DictionaryType;
  typedef itk::MetaDataObject< std::string > MetaDataStringType;

  // Check if the DICOM Inverse tag is set

  std::string tagInverse = "2050|0020";
  
  DictionaryType::ConstIterator tagInverseItr = dictionary.Find( tagInverse );
  DictionaryType::ConstIterator tagInverseEnd = dictionary.End();
  
  if ( tagInverseItr != tagInverseEnd )
  {
    MetaDataStringType::ConstPointer entryvalue = 
      dynamic_cast<const MetaDataStringType *>( tagInverseItr->second.GetPointer() );
    
    if ( entryvalue )
    {
      std::string strInverse( "INVERSE" );
      std::string tagInverseValue = entryvalue->GetMetaDataObjectValue();
      
      std::cout << "Tag (" << tagInverse 
		<< ") is: " << tagInverseValue << std::endl;

      std::size_t foundInverse = tagInverseValue.find( strInverse );
      if (foundInverse != std::string::npos)
      {
	flgInvert = true;
	std::cout << "Image is INVERSE - inverting" << std::endl;
	SetTag( dictionary, tagInverse, "IDENTITY" );
      }
    }
  }


  // Fix the MONOCHROME1 issue

  std::string tagPhotoInterpID = "0028|0004";
  
  DictionaryType::ConstIterator tagPhotoInterpItr = dictionary.Find( tagPhotoInterpID );
  DictionaryType::ConstIterator tagPhotoInterpEnd = dictionary.End();
  
  if ( tagPhotoInterpItr != tagPhotoInterpEnd )
  {
    MetaDataStringType::ConstPointer entryvalue = 
      dynamic_cast<const MetaDataStringType *>( tagPhotoInterpItr->second.GetPointer() );
    
    if ( entryvalue )
    {
      std::string strMonochrome1( "MONOCHROME1" );
      std::string tagPhotoInterpValue = entryvalue->GetMetaDataObjectValue();
      
      std::cout << "Tag (" << tagPhotoInterpID 
		<< ") is: " << tagPhotoInterpValue << std::endl;

      std::size_t foundMonochrome1 = tagPhotoInterpValue.find( strMonochrome1 );
      if (foundMonochrome1 != std::string::npos)
      {
	flgInvert = true;
	std::cout << "Image is MONOCHROME1 - inverting" << std::endl;
	SetTag( dictionary, tagPhotoInterpID, "MONOCHROME2" );
      }
    }
  }


  // Invert the image

  if ( flgInvert )
  {
    typedef typename itk::InvertIntensityBetweenMaxAndMinImageFilter< TImage > InvertFilterType;
    
    typename InvertFilterType::Pointer invertFilter = InvertFilterType::New();
    invertFilter->SetInput( image );

    invertFilter->Update( );
	
    image = invertFilter->GetOutput();
    image->DisconnectPipeline();
  }

};


//  --------------------------------------------------------------------------
/// Convert a raw DICOM mammogram to a presentation version by log inverting it
//  --------------------------------------------------------------------------

template < typename TImage >
bool ConvertMammogramFromRawToPresentation( typename TImage::Pointer &image,
                                            itk::MetaDataDictionary &dictionary )
{
  bool flgPreInvert = false;

  itksys_ios::ostringstream value;

  typedef itk::MetaDataDictionary DictionaryType;
  typedef itk::MetaDataObject< std::string > MetaDataStringType;

  typedef itk::MinimumMaximumImageCalculator< TImage > MinimumMaximumImageCalculatorType;
  typedef itk::RescaleIntensityImageFilter< TImage, TImage > RescalerType;
  typedef itk::InvertIntensityBetweenMaxAndMinImageFilter< TImage > InvertFilterType;
  typedef itk::LogNonZeroIntensitiesImageFilter< TImage, TImage > LogFilterType;

  DictionaryType::ConstIterator tagItr;
  DictionaryType::ConstIterator tagEnd;

  // Check that the modality DICOM tag is 'MG'

  std::string tagModalityID = "0008|0060";
  std::string tagModalityValue;

  tagItr = dictionary.Find( tagModalityID );
  tagEnd = dictionary.End();
   
  if( tagItr != tagEnd )
  {
    MetaDataStringType::ConstPointer entryvalue = 
      dynamic_cast<const MetaDataStringType *>( tagItr->second.GetPointer() );

    if ( entryvalue )
    {
      tagModalityValue = entryvalue->GetMetaDataObjectValue();
      std::cout << "Modality Name (" << tagModalityID <<  ") "
                << " is: " << tagModalityValue << std::endl;
    }
  }

  // Check that the 'Presentation Intent Type' is 'For Processing'

  std::string tagForProcessingID = "0008|0068";
  std::string tagForProcessingValue;

  tagItr = dictionary.Find( tagForProcessingID );
  tagEnd = dictionary.End();
   
  if( tagItr != tagEnd )
  {
    MetaDataStringType::ConstPointer entryvalue = 
      dynamic_cast<const MetaDataStringType *>( tagItr->second.GetPointer() );

    if ( entryvalue )
    {
      tagForProcessingValue = entryvalue->GetMetaDataObjectValue();
      std::cout << "Presentation Intent Type (" << tagForProcessingID <<  ") "
                << " is: " << tagForProcessingValue << std::endl;
    }
  }

  // Process this file?

  if ( ( ( tagModalityValue == std::string( "CR" ) ) ||    //  Computed Radiography
         ( tagModalityValue == std::string( "MG" ) ) ) &&  //  Mammography
       ( tagForProcessingValue == std::string( "FOR PROCESSING" ) ) )
  {
    std::cout << "Image is a raw \"FOR PROCESSING\" mammogram - converting"
              << std::endl;
  }
  else
  {
    std::cout << "Skipping image - does not appear to be a \"FOR PROCESSING\" mammogram" 
              << std::endl << std::endl;
    return false;
  }

  // Set the desired output range (i.e. the same as the input)

  typename MinimumMaximumImageCalculatorType::Pointer 
    imageRangeCalculator = MinimumMaximumImageCalculatorType::New();

  imageRangeCalculator->SetImage( image );
  imageRangeCalculator->Compute();


  // Change the tag to "FOR PRESENTATION"

  ModifyTag( dictionary, "0008|0068", "FOR PRESENTATION" );

  // Set the pixel intensity relationship sign to linear
  value.str("");
  value << "LIN";
  itk::EncapsulateMetaData<std::string>(dictionary,"0028|1040", value.str());

  // Set the pixel intensity relationship sign to one
  value.str("");
  value << 1;
  itk::EncapsulateMetaData<std::string>(dictionary,"0028|1041", value.str());

  // Set the presentation LUT shape
  ModifyTag( dictionary, "2050|0020", "IDENTITY" );
    
  // Check whether this is MONOCHROME1 or 2 and hence whether to invert

  std::string tagPhotoInterpID = "0028|0004";
  std::string tagPhotoInterpValue;

  tagItr = dictionary.Find( tagPhotoInterpID );
  tagEnd = dictionary.End();
   
  if( tagItr != tagEnd )
  {
    MetaDataStringType::ConstPointer entryvalue = 
      dynamic_cast<const MetaDataStringType *>( tagItr->second.GetPointer() );

    if ( entryvalue )
    {
      tagPhotoInterpValue = entryvalue->GetMetaDataObjectValue();
      std::cout << "Photometric interportation is (" << tagPhotoInterpID <<  ") "
                << " is: " << tagPhotoInterpValue << std::endl;
    }
  }

  std::size_t found = tagPhotoInterpValue.find( "MONOCHROME2" );
  if ( found != std::string::npos )
  {
    std::cout << "Image is \"MONOCHROME2\" so will not be inverted"
              << std::endl;
    flgPreInvert = true;        // Actually we pre-invert it
  }

  found = tagPhotoInterpValue.find( "MONOCHROME1" );
  if ( found != std::string::npos )
  {
    ModifyTag( dictionary, "0028|0004", "MONOCHROME2" );
  }

   
  // Convert the image to a "FOR PRESENTATION" version by calculating the logarithm and inverting 

  if ( flgPreInvert ) 
  {
    typename InvertFilterType::Pointer invfilter = InvertFilterType::New();
    invfilter->SetInput( image );
    invfilter->UpdateLargestPossibleRegion();
    image = invfilter->GetOutput();
  }

  typename LogFilterType::Pointer logfilter = LogFilterType::New();
  logfilter->SetInput(image);
  logfilter->UpdateLargestPossibleRegion();
   
  typename InvertFilterType::Pointer invfilter = InvertFilterType::New();
  invfilter->SetInput(logfilter->GetOutput());
  invfilter->UpdateLargestPossibleRegion();

  image = invfilter->GetOutput();
  image->DisconnectPipeline();


  // Rescale the image

  typename RescalerType::Pointer intensityRescaler = RescalerType::New();

  intensityRescaler->SetOutputMinimum( 
    static_cast< typename TImage::PixelType >( imageRangeCalculator->GetMinimum() ) );
  intensityRescaler->SetOutputMaximum( 
    static_cast< typename TImage::PixelType >( imageRangeCalculator->GetMaximum() ) );

  std::cout << "Image output range will be: " << intensityRescaler->GetOutputMinimum()
            << " to " << intensityRescaler->GetOutputMaximum() << std::endl;

  intensityRescaler->SetInput( image );  

  intensityRescaler->UpdateLargestPossibleRegion();

  image = intensityRescaler->GetOutput();
  image->DisconnectPipeline();



  return true;                  // This is a raw mammogram that has been converted
}

} // end namespace itk

#endif /* __itkCreatePositiveMammogram_h */
