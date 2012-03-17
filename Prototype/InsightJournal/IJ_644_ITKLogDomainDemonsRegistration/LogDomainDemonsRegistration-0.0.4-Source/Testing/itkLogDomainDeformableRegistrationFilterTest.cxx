#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#include <iostream>

#include "itkLogDomainDeformableRegistrationFilter.h"


/* This test does nothing since LogDomainDeformableRegistrationFilter
 * is a base class. The sole purpose of this file is to test potential
 * compilation issues. */

int main(int, char* [] )
{
  const unsigned int ImageDimension = 2;

  typedef itk::Vector<float,ImageDimension> VectorType;
  typedef itk::Image<VectorType,ImageDimension> FieldType;
  typedef itk::Image<float,ImageDimension> ImageType;

  typedef FieldType::PixelType  PixelType;
  typedef FieldType::IndexType  IndexType;

  typedef itk::LogDomainDeformableRegistrationFilter<ImageType,ImageType,FieldType> FilterType;
  FilterType::Pointer filter = FilterType::New();


  std::cout << "Test passed." << std::endl;
  return EXIT_SUCCESS;
}
