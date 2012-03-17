#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#include <iostream>

#include "itkVector.h"
#include "itkIndex.h"
#include "itkImage.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkVectorCastImageFilter.h"
#include "itkVelocityFieldLieBracketFilter.h"
#include "itkSmoothingRecursiveGaussianImageFilter.h"
#include "itkStreamingImageFilter.h"
#include "itkAddImageFilter.h"
#include "itkMultiplyByConstantImageFilter.h"
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

  std::cout << "Run VelocityFieldLieBracketFilter in standalone mode with progress.";
  std::cout << std::endl;

  typedef itk::VelocityFieldLieBracketFilter<FieldType,FieldType> ComposerType;
  ComposerType::Pointer composer = ComposerType::New();

  composer->SetInput( 0, leftfield );
  composer->SetInput( 1, leftfield ); // --> To check that [u,u] == 0

  ShowProgressObject progressWatch(composer);
  itk::SimpleMemberCommand<ShowProgressObject>::Pointer command;
  command = itk::SimpleMemberCommand<ShowProgressObject>::New();
  command->SetCallbackFunction(&progressWatch,
                               &ShowProgressObject::ShowProgress);
  composer->AddObserver(itk::ProgressEvent(), command);

  composer->Print( std::cout );

  // exercise Get methods

  // exercise Set methods
 
  // Update the filter
  composer->Update();

  // Remove progress reporter
  composer->RemoveAllObservers();
  

  //=============================================================

  std::cout << "Checking the output against expected." << std::endl;

  std::cout << "1) Checking that [u,u] = 0." <<std::endl;

  FieldIterator outComposerIter( composer->GetOutput(),
                                 composer->GetOutput()->GetBufferedRegion() );
  
  outComposerIter.GoToBegin();
  while( !outComposerIter.IsAtEnd() )
   {
    if( ( outComposerIter.Get() ).GetNorm() != static_cast<PixelType::RealValueType>(0) )
      {
       testPassed = false;
       std::cout << "Failed: [u,u] != 0."<< std::endl;
       break;
      }
    
    ++outComposerIter;
   }

  // ------------------------------------
  
  std::cout << "2) Checking that [u,v] = -[v,u]." << std::endl;

  ComposerType::Pointer composerBis = ComposerType::New();
  
  // Computing [u,v]
  composer->SetInput( 1, rightfield );
  composer->Update();

  // Computing [v,u]
  composerBis->SetInput( 0, rightfield );
  composerBis->SetInput( 1, leftfield );
  composerBis->Update();
  
  FieldIterator uvComposerIter( composer->GetOutput(),
                      composer->GetOutput()->GetBufferedRegion() );
  FieldIterator vuComposerIter( composerBis->GetOutput(),
                      composerBis->GetOutput()->GetBufferedRegion() );
  
  // Checking that [u,v] + [v,u] = 0
  unsigned int nbPixel = 0;
  double squareDiff = 0.0;
  double mean = 0.0;
  uvComposerIter.GoToBegin();
  vuComposerIter.GoToBegin();

  while( !uvComposerIter.IsAtEnd() )
   {
    const FieldType::PixelType & uvVal = uvComposerIter.Get();
    const FieldType::PixelType & vuVal = vuComposerIter.Get();

    squareDiff += (uvVal + vuVal).GetSquaredNorm();

    ++uvComposerIter;
    ++vuComposerIter;
    ++nbPixel;
   }

  mean = squareDiff/(double)nbPixel;

  if ( mean > 1e-6 )
   {
    testPassed = false;
    std::cout.precision(6);
    std::cout << "Failed. Error: " << mean << std::endl;
   }
  

  // ------------------------------------

  std::cout << "3) Checking that [ku + u',v] = k[u,v] + [u',v]." << std::endl;

  double k = 5.0;
  
  // Generate another deformation filter
  FieldType::RegionType leftregion2( leftregion );
  FieldType::SizeType leftsize2( leftsize );
  leftregion2.SetSize( leftsize2 );

  FieldType::Pointer leftfield2 = FieldType::New();
  // Set LargestPossibleRegion, BufferedRegion, and RequestedRegion simultaneously.
  leftfield2->SetRegions( leftregion2 );
  leftfield2->Allocate();
  
  // Fill the field with random values
  FieldIterator leftIter2( leftfield2, leftfield2->GetRequestedRegion() );
  
  for ( leftIter2.GoToBegin(); !leftIter2.IsAtEnd(); ++leftIter2 )
    {
    PixelType & value = leftIter2.Value();
    for ( unsigned int  i=0; i<ImageDimension; ++i )
      {
       value[i] = power * rng.normal();
      }
    }

  // Smooth the field
  smootherX->SetInput( leftfield2 );
  smootherY->SetInput( smootherX->GetOutput() );
  smootherY->Update();
  leftfield2 = smootherY->GetOutput();
  leftfield2->DisconnectPipeline();

  // Adder
  typedef itk::AddImageFilter<FieldType,FieldType>  AdderType;
  AdderType::Pointer adder = AdderType::New();
  AdderType::Pointer adder2 = AdderType::New();

  // Multiplier
  typedef itk::MultiplyByConstantImageFilter<FieldType,double,FieldType>   MultiplyByConstantType;
  MultiplyByConstantType::Pointer multiplier = MultiplyByConstantType::New();
  MultiplyByConstantType::Pointer multiplier2 = MultiplyByConstantType::New();
 
  // Composer
  ComposerType::Pointer composerTer = ComposerType::New();

  // Compute [ku + u',v]
  multiplier->SetConstant( k );
  multiplier->SetInput( leftfield );
  
  adder->SetInput1( multiplier->GetOutput() );
  adder->SetInput2( leftfield2 );

  composer->SetInput( 0, adder->GetOutput() );
  composer->SetInput( 1, rightfield );
  composer->Update();

  // Compute k[u,v] + [u',v]
  composerBis->SetInput( 0, leftfield );
  composerBis->SetInput( 1, rightfield );

  multiplier2->SetConstant( k );
  multiplier2->SetInput( composerBis->GetOutput() );

  composerTer->SetInput( 0, leftfield2);
  composerTer->SetInput( 1, rightfield );

  adder2->SetInput1( multiplier2->GetOutput() );
  adder2->SetInput2( composerTer->GetOutput() );
  adder2->Update();

  // Check that [ku + u',v] - k[u,v] + [u',v] = 0
  FieldIterator iter1( composer->GetOutput(),
                composer->GetOutput()->GetBufferedRegion() );
  FieldIterator iter2( adder2->GetOutput(),
                adder2->GetOutput()->GetBufferedRegion() );

  iter1.GoToBegin();
  iter2.GoToBegin();
  mean = 0.0;
  squareDiff = 0.0;

  while( !iter1.IsAtEnd() )
   {
    const FieldType::PixelType & val1 = iter1.Get();
    const FieldType::PixelType & val2 = iter2.Get();

    squareDiff += (val1 - val2).GetSquaredNorm();

    ++iter1;
    ++iter2;
   }

  mean = squareDiff/(double)nbPixel;

  if ( mean > 1e-6 )
   {
    testPassed = false;
    std::cout.precision(6);
    std::cout << "Failed. Error: " << mean << std::endl;
   }


  // ------------------------------------
  {
  std::cout << "4) Checking Jacobi identity [u,[v,w]] + [w,[u,v]] + [v,[w,u]] = 0." << std::endl;

  FieldType::Pointer u_field = leftfield;
  FieldType::Pointer v_field = rightfield;
  FieldType::Pointer w_field = leftfield2;
                                     
  // [u,[v,w]]
  ComposerType::Pointer vw_comp = ComposerType::New();
  vw_comp->SetInput( 0, v_field );
  vw_comp->SetInput( 1, w_field );

  ComposerType::Pointer uvw_comp = ComposerType::New();
  uvw_comp->SetInput( 0, u_field );
  uvw_comp->SetInput( 1, vw_comp->GetOutput() );
  uvw_comp->UpdateLargestPossibleRegion();
  FieldType::Pointer uvw_field = uvw_comp->GetOutput();

  // [w,[u,v]]
  ComposerType::Pointer uv_comp = ComposerType::New();
  uv_comp->SetInput( 0, u_field );
  uv_comp->SetInput( 1, v_field );

  ComposerType::Pointer wuv_comp = ComposerType::New();
  wuv_comp->SetInput( 0, w_field );
  wuv_comp->SetInput( 1, uv_comp->GetOutput() );
  wuv_comp->UpdateLargestPossibleRegion();
  FieldType::Pointer wuv_field = wuv_comp->GetOutput();


  // [v,[w,u]]
  ComposerType::Pointer wu_comp = ComposerType::New();
  wu_comp->SetInput( 0, w_field );
  wu_comp->SetInput( 1, u_field );

  ComposerType::Pointer vwu_comp = ComposerType::New();
  vwu_comp->SetInput( 0, v_field );
  vwu_comp->SetInput( 1, wu_comp->GetOutput() );
  vwu_comp->UpdateLargestPossibleRegion();
  FieldType::Pointer vwu_field = vwu_comp->GetOutput();


  // [u,[v,w]] + [w,[u,v]] + [v,[w,u]]
  FieldIterator uvw_iter( uvw_field, uvw_field->GetLargestPossibleRegion() );
  FieldIterator wuv_iter( wuv_field, wuv_field->GetLargestPossibleRegion() );
  FieldIterator vwu_iter( vwu_field, vwu_field->GetLargestPossibleRegion() );

  double uvw_energy(0.0), wuv_energy(0.0), vwu_energy(0.0), jacobi_energy(0.0);

  while ( !uvw_iter.IsAtEnd() )
    {
    uvw_energy += uvw_iter.Value().GetSquaredNorm();
    wuv_energy += wuv_iter.Value().GetSquaredNorm();
    vwu_energy += vwu_iter.Value().GetSquaredNorm();
    
    const PixelType jacobi_pix = uvw_iter.Value() + wuv_iter.Value() + vwu_iter.Value();
    jacobi_energy += jacobi_pix.GetSquaredNorm();
    
    ++uvw_iter;
    ++wuv_iter;
    ++vwu_iter;
    }

  uvw_energy = std::sqrt( uvw_energy / (double)nbPixel );
  wuv_energy = std::sqrt( wuv_energy / (double)nbPixel );
  vwu_energy = std::sqrt( vwu_energy / (double)nbPixel );
  jacobi_energy = std::sqrt( jacobi_energy / (double)nbPixel );
  const double compared_energy = jacobi_energy / (uvw_energy + wuv_energy + vwu_energy);

  std::cout << "  uvw_energy:    " << uvw_energy << std::endl;
  std::cout << "  wuv_energy:    " << wuv_energy << std::endl;
  std::cout << "  vwu_energy:    " << vwu_energy << std::endl;
  std::cout << "  jacobi_energy: " << jacobi_energy << std::endl;
  std::cout << "  j_e/(uvw_e+wuv_e+vwu_e): " << compared_energy << std::endl;

  // Right now the test is very easy to passs
  // It seems that the central difference scheme doen't play well
  // with imbricated Lie brackets...
  if ( compared_energy >= 1e-1 )
    {
    testPassed = false;
    std::cout << "Failed: [u,[v,w]] + [w,[u,v]] + [v,[w,u]] != 0."<< std::endl;
    }

  }


  //=============================================================

  std::cout << "Run Filter with streamer";
  std::cout << std::endl;

  typedef itk::VectorCastImageFilter<FieldType,FieldType> VectorCasterType;
  VectorCasterType::Pointer vcaster = VectorCasterType::New();

  vcaster->SetInput( composer->GetInput(1) );

  ComposerType::Pointer composer2 = ComposerType::New();

  composer2->SetInput( 0, composer->GetInput(0) );
  composer2->SetInput( 1, vcaster->GetOutput() );

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

  while( !outIter.IsAtEnd() )
    {
    if( outIter.Get() != streamIter.Get() )
      {
       testPassed = false;
       std::cout << "Streaming failed." << std::endl;
       break;
      }
    ++outIter;
    ++streamIter;
    }
  

  // Exercise error handling
 //  typedef ComposerType::WarpGradientCalculatorType gradientCalculatorType;
//   gradientCalculatorType::Pointer leftGradCalculator = composer->GetLeftGradientCalculator();
//   gradientCalculatorType::Pointer rightGradCalculator = composer->GetRightGradientCalculator();
 
//   try
//     {
//     std::cout << "Setting gradient calculators to NULL" << std::endl;
//     testPassed = false;
//   composer->SetLeftGradientCalculator( NULL );
//     composer->SetRightGradientCalculator( NULL );
//   composer->Update();
//     }
//   catch( itk::ExceptionObject& err )
//     {
//     std::cout << err << std::endl;
//     testPassed = true;
//     composer->ResetPipeline();
//   composer->SetLeftGradientCalculator( leftGradCalculator );
//   composer->SetRightGradientCalculator( rightGradCalculator );
//     }

//   if (!testPassed) {
//     std::cout << "Test failed" << std::endl;
//     return EXIT_FAILURE;
//     }


  if ( !testPassed )
    {
    std::cout << "Test failed." << std::endl;
    return EXIT_FAILURE;
    }

  std::cout << "Test passed." << std::endl;
  return EXIT_SUCCESS;

}
