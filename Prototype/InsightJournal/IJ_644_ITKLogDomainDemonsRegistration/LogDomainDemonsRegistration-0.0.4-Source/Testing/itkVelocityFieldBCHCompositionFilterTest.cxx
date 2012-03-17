#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#include <iostream>

#include "itkVector.h"
#include "itkIndex.h"
#include "itkImage.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkVectorCastImageFilter.h"
#include "itkExponentialDeformationFieldImageFilter2.h"
#include "itkDisplacementFieldCompositionFilter.h"
#include "itkSmoothingRecursiveGaussianImageFilter.h"
#include "itkVelocityFieldBCHCompositionFilter.h"
#include "itkStreamingImageFilter.h"
#include "vnl/vnl_math.h"
#include <vnl/vnl_random.h>
#include "itkCommand.h"



// The following three classes are used to support callbacks
// on the filter in the pipeline that follows later
class ShowProgressObject
{
public:
  ShowProgressObject(itk::ProcessObject* o)
  {m_Process = o;}
  void ShowProgress()
  {std::cout << "Progress " << m_Process->GetProgress() << std::endl;}
  itk::ProcessObject::Pointer m_Process;
};



int main(int, char* [] )
{
  const unsigned int ImageDimension = 2;

  typedef itk::Vector<float,ImageDimension> VectorType;
  typedef itk::Image<VectorType,ImageDimension> FieldType;

  typedef FieldType::PixelType  PixelType;
  typedef FieldType::IndexType  IndexType;

  typedef itk::ImageRegionIteratorWithIndex<FieldType> FieldIterator;
  
  bool testPassed = true;

  // Random number generator
  vnl_random rng;
  const double power = 5.0;

  //=============================================================

  std::cout << "Create the left deformation field." << std::endl;
  
  FieldType::RegionType leftregion;
  FieldType::SizeType leftsize = {{64, 64}};
  leftregion.SetSize( leftsize );
  
  FieldType::Pointer leftfield = FieldType::New();
  // Set LargestPossibleRegion, BufferedRegion, and RequestedRegion simultaneously.
  leftfield->SetRegions( leftregion );
  leftfield->Allocate();

  // Fill the field with random values
  FieldIterator leftIter( leftfield, leftfield->GetRequestedRegion() );
  
  for ( leftIter.GoToBegin(); !leftIter.IsAtEnd(); ++leftIter )
    {
    PixelType & value = leftIter.Value();
    for ( unsigned int  i=0; i<ImageDimension; ++i )
      {
       value[i] = power * rng.normal();
      }
    }

  
  //=============================================================

  std::cout << "Create the right deformation field." << std::endl;

  FieldType::RegionType rightregion( leftregion );
  FieldType::SizeType rightsize( leftsize );
  rightregion.SetSize( rightsize );

  FieldType::Pointer rightfield = FieldType::New();
  // Set LargestPossibleRegion, BufferedRegion, and RequestedRegion simultaneously.
  rightfield->SetRegions( rightregion );
  rightfield->Allocate();
  
  // Fill the field with random values
  FieldIterator rightIter( rightfield, rightfield->GetRequestedRegion() );
  
  for ( rightIter.GoToBegin(); !rightIter.IsAtEnd(); ++rightIter )
    {
    PixelType & value = rightIter.Value();
    for ( unsigned int  i=0; i<ImageDimension; ++i )
      {
       value[i] = power * rng.normal();
      }
    }
  

  //=============================================================

  std::cout << "Smooth left and right fields." << std::endl;

  typedef itk::RecursiveGaussianImageFilter< FieldType, FieldType > smootherType;

  smootherType::Pointer smootherX = smootherType::New();
  smootherType::Pointer smootherY = smootherType::New();

  smootherType::ScalarRealType sigma = 2.0;

  smootherX->SetDirection( 0 ); // 0 --> X direction
  smootherY->SetDirection( 1 ); // 1 --> Y direction

  // Set the order 
  smootherX->SetOrder( smootherType::ZeroOrder );
  smootherY->SetOrder( smootherType::ZeroOrder );

  smootherX->SetNormalizeAcrossScale( false );
  smootherY->SetNormalizeAcrossScale( false );

  // Smooth left field
  smootherX->SetInput( leftfield );
  smootherY->SetInput( smootherX->GetOutput() );

  // Set sigma
  smootherX->SetSigma( sigma );
  smootherY->SetSigma( sigma );
  
  // Trigger update
  smootherY->Update();

  // Re-affect leftfield
  leftfield = smootherY->GetOutput();
  leftfield->DisconnectPipeline();

  // Smooth right field
  smootherX->SetInput( rightfield );
  smootherY->SetInput( smootherX->GetOutput() );

  // Trigger update
  smootherY->Update();

  // Re-affect rightfield
  rightfield = smootherY->GetOutput();
  rightfield->DisconnectPipeline();


  //=============================================================


  std::cout << "Run VelocityFieldBCHCompositionFilter in standalone mode with progress.";
  std::cout << std::endl;

  typedef itk::VelocityFieldBCHCompositionFilter<FieldType,FieldType> ComposerType;
  ComposerType::Pointer composer = ComposerType::New();

  composer->SetInput( 0, leftfield );
  composer->SetInput( 1, rightfield );

  ShowProgressObject progressWatch(composer);
  itk::SimpleMemberCommand<ShowProgressObject>::Pointer command;
  command = itk::SimpleMemberCommand<ShowProgressObject>::New();
  command->SetCallbackFunction(&progressWatch,
                               &ShowProgressObject::ShowProgress);
  composer->AddObserver(itk::ProgressEvent(), command);

  composer->Print( std::cout );

  // exercise Get methods

  // exercise Set methods
  

  //=============================================================

  std::cout << "Checking the output against expected." << std::endl;
  
  std::cout << "1) Compute the exponential of those velocity fields." << std::endl;

  // Exponential calculator filter type
  typedef itk::ExponentialDeformationFieldImageFilter<
  FieldType, FieldType >   ExponentiatorType;
    
  ExponentiatorType::Pointer exponentiator = ExponentiatorType::New();
  ExponentiatorType::Pointer exponentiator2 = ExponentiatorType::New();

  // Set the inputs
  exponentiator->SetInput( leftfield );
  exponentiator2->SetInput( rightfield );
  
  std::cout << "Compose the deformation fields computed." << std::endl;

  // Field composer type
  typedef itk::DisplacementFieldCompositionFilter<
  FieldType, FieldType >   fieldComposerType;

  fieldComposerType::Pointer fieldComposer = fieldComposerType::New();

  // Set the inputs
  fieldComposer->SetInput( 0, exponentiator->GetOutput() );
  fieldComposer->SetInput( 1, exponentiator2->GetOutput() );

  // Trigger update
  fieldComposer->Update();


  //=============================================================

  /* Compare the accuracy of different BCH approximations by
     comparing the error between the exponential of the BCH
     approximations and the actual composition.
  */

  std::cout << "2) Compare the accuracy of different BCH approximations." << std::endl;

  ExponentiatorType::Pointer exponentiatorBCH = ExponentiatorType::New();
  exponentiatorBCH->SetInput( composer->GetOutput() );

  FieldIterator outComposerIter( fieldComposer->GetOutput(),
                                 fieldComposer->GetOutput()->GetBufferedRegion() );

  std::vector<double> MSEs(3,0.0);
  for (unsigned int num_bch_terms=2; num_bch_terms<5; ++num_bch_terms)
    {
    std::cout << "Compute with number of terms = "<< num_bch_terms << std::endl;
    
    // Set order for BCH formula
    composer->SetNumberOfApproximationTerms( num_bch_terms );
    
    // Compute exponential of BCH output
    exponentiatorBCH->Update();
    
    
    FieldIterator outBCHIter( exponentiatorBCH->GetOutput(),
                      exponentiatorBCH->GetOutput()->GetBufferedRegion() );
    
    unsigned int nbPixel = 0;
    double squareDiff = 0.0;
    outBCHIter.GoToBegin();
    outComposerIter.GoToBegin();
    
    while( !outBCHIter.IsAtEnd() )
      {
       const FieldType::PixelType & bchVal = outBCHIter.Get();
       const FieldType::PixelType & composeVal = outComposerIter.Get();
       
       squareDiff += (bchVal - composeVal).GetSquaredNorm();
       
       ++outBCHIter;
       ++outComposerIter;
       ++nbPixel;
      }

    MSEs[num_bch_terms-2] = squareDiff/(double)nbPixel;
    std::cout << "Error at order " << num_bch_terms << ": " << MSEs[num_bch_terms-2] << std::endl;


    //=============================================================
    
    std::cout << "Run Filter with streamer";
    std::cout << std::endl;
    
    typedef itk::VectorCastImageFilter<FieldType,FieldType> VectorCasterType;
    VectorCasterType::Pointer vcaster = VectorCasterType::New();
    
    vcaster->SetInput( composer->GetInput(1) );
    
    ComposerType::Pointer composer2 = ComposerType::New();
    
    composer2->SetInput( 0, composer->GetInput(0) );
    composer2->SetInput( 1, vcaster->GetOutput() );
    composer2->SetNumberOfApproximationTerms( composer->GetNumberOfApproximationTerms() );
    
    typedef itk::StreamingImageFilter<FieldType,FieldType> StreamerType;
    StreamerType::Pointer streamer = StreamerType::New();
    streamer->SetInput( composer2->GetOutput() );
    streamer->SetNumberOfStreamDivisions( 3 );
    streamer->Update();

  
    //=============================================================
    
    std::cout << "Compare standalone and streamed outputs" << std::endl;
    
    FieldIterator outIter( composer->GetOutput(),
                    composer->GetOutput()->GetBufferedRegion() );
    
    FieldIterator streamIter( streamer->GetOutput(),
                      streamer->GetOutput()->GetBufferedRegion() );
    
    outIter.GoToBegin();
    streamIter.GoToBegin();
    bool streamTestPassed = true;

    while( !outIter.IsAtEnd() )
      {
       if( outIter.Get() != streamIter.Get() )
        {
          streamTestPassed = false;
        }
       ++outIter;
       ++streamIter;
      }
    
    if ( !streamTestPassed )
      {
       testPassed = false;
       std::cout << "Streaming failed with number of terms = " << num_bch_terms << std::endl;
      }
    }

  
  std::cout << std::endl;
  std::cout << "Comparing errors."<< std::endl;
  std::cout << "MSE with number of terms = " << 2 << " :  MSE = " << MSEs[0] << std::endl;

  for (unsigned int num_bch_terms=3; num_bch_terms<5; ++num_bch_terms)
    {
    if ( MSEs[num_bch_terms-2]>MSEs[num_bch_terms-3] ) testPassed = false;
    std::cout << "MSE with number of terms = " << num_bch_terms << ": "
              << MSEs[num_bch_terms-2] << std::endl;
    }
  

  if ( !testPassed )
    {
    std::cout << "Test failed." << std::endl;
    return EXIT_FAILURE;
    }

  // Exercise error handling
 

  std::cout << "Test passed." << std::endl;
  return EXIT_SUCCESS;

}
