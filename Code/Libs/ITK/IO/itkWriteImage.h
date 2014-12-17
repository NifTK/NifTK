/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkWriteImage_h
#define __itkWriteImage_h

#include <itkImageFileWriter.h>


namespace itk
{



//  --------------------------------------------------------------------------
/// Write an ITK image to a file and print a message
//  --------------------------------------------------------------------------

template < typename TOutputImage >
void
WriteImageToFile( const char *fileOutput, const char *description,
                  typename TOutputImage::Pointer image )
{
  if ( fileOutput ) {

    typedef itk::ImageFileWriter< TOutputImage > FileWriterType;

    typename FileWriterType::Pointer writer = FileWriterType::New();

    writer->SetFileName( fileOutput );
    writer->SetInput( image );

    std::cout << "Writing " << description << " to file: "
              << fileOutput << std::endl;

    writer->Update();
  }
}

//  --------------------------------------------------------------------------
/// Write an ITK image to a file and print a message
//  --------------------------------------------------------------------------

template < typename TOutputImage >
void
WriteImageToFile( const char *fileOutput, const char *description,
                  typename TOutputImage::ConstPointer image )
{
  if ( fileOutput ) {

    typedef itk::ImageFileWriter< TOutputImage > FileWriterType;

    typename FileWriterType::Pointer writer = FileWriterType::New();

    writer->SetFileName( fileOutput );
    writer->SetInput( image );

    std::cout << "Writing " << description << " to file: "
              << fileOutput << std::endl;

    writer->Update();
  }
}


//  --------------------------------------------------------------------------
/// Write an ITK image to a file and print a message
//  --------------------------------------------------------------------------

template < typename TOutputImage >
void
WriteImageToFile( std::string fileOutput, typename TOutputImage::Pointer image, 
                  std::string *description=0 )
{
  if ( fileOutput.length() > 0 ) {

    typedef itk::ImageFileWriter< TOutputImage > FileWriterType;

    typename FileWriterType::Pointer writer = FileWriterType::New();

    writer->SetFileName( fileOutput );
    writer->SetInput( image );

    if ( description )
    {
      std::cout << "Writing " << description << " to file: "
                << fileOutput << std::endl;
    }

    writer->Update();
  }
}

//  --------------------------------------------------------------------------
/// Write an ITK image to a file and print a message
//  --------------------------------------------------------------------------

template < typename TOutputImage >
void
WriteImageToFile( std::string fileOutput, typename TOutputImage::ConstPointer image, 
                  std::string *description=0 )
{
  if ( fileOutput.length() > 0 ) {

    typedef itk::ImageFileWriter< TOutputImage > FileWriterType;

    typename FileWriterType::Pointer writer = FileWriterType::New();

    writer->SetFileName( fileOutput );
    writer->SetInput( image );

    if ( description )
    {
      std::cout << "Writing " << description << " to file: "
                << fileOutput << std::endl;
    }

    writer->Update();
  }
}




} // end namespace itk

#endif /* __itkWriteImage_h */
