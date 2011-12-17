#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#include <iostream>

#include "itkVector.h"
#include "itkIndex.h"
#include "itkImage.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkDisplacementFieldCompositionFilter.h"
#include "itkVectorCastImageFilter.h"
#include "itkStreamingImageFilter.h"
#include "vnl/vnl_math.h"
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


  //=============================================================

  std::cout << "Create the left deformation field." << std::endl;
  
  FieldType::RegionType leftregion;
  FieldType::SizeType leftsize = {{64, 64}};
  leftregion.SetSize( leftsize );
  
  FieldType::Pointer leftfield = FieldType::New();
  leftfield->SetLargestPossibleRegion( leftregion );
  leftfield->SetBufferedRegion( leftregion );
  leftfield->Allocate();

  
  //=============================================================

  std::cout << "Create the right deformation field." << std::endl;

  FieldType::RegionType rightregion( leftregion );
  FieldType::SizeType rightsize( leftsize );
  rightregion.SetSize( rightsize );

  FieldType::Pointer rightfield = FieldType::New();
  rightfield->SetLargestPossibleRegion( rightregion );
  rightfield->SetBufferedRegion( rightregion );
  rightfield->Allocate();


  //=============================================================

  std::cout << "Run DisplacementFieldCompositionFilter in standalone mode with progress.";
  std::cout << std::endl;
  typedef itk::DisplacementFieldCompositionFilter<FieldType,FieldType> ComposerType;
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
 
  // Update the filter
  composer->Update();
  

  //=============================================================

  std::cout << "Checking the output against expected." << std::endl;
  

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
      }
    ++outIter;
    ++streamIter;
    }
  

  if ( !testPassed )
    {
    std::cout << "Test failed." << std::endl;
    return EXIT_FAILURE;
    }

  // Exercise error handling
  typedef ComposerType::VectorWarperType VectorWarperType;
  VectorWarperType::Pointer warper = composer->GetWarper();
 
  try
    {
    std::cout << "Setting warper to NULL" << std::endl;
    testPassed = false;
    composer->SetWarper( NULL );
    composer->Update();
    }
  catch( itk::ExceptionObject& err )
    {
    std::cout << err << std::endl;
    testPassed = true;
    composer->ResetPipeline();
    composer->SetWarper( warper );
    }

  if (!testPassed) {
    std::cout << "Test failed" << std::endl;
    return EXIT_FAILURE;
    }

 std::cout << "Test passed." << std::endl;
 return EXIT_SUCCESS;

}
