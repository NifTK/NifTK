/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkReadImage_h
#define __itkReadImage_h

#include <itkImageFileReader.h>

#include <niftkFileHelper.h>


namespace itk
{



//  --------------------------------------------------------------------------
/// Read an ITK image to a file and print a message
//  --------------------------------------------------------------------------

template < typename TInputImage >
bool
ReadImageFromFile( const char *fileInput, const char *description,
                   typename TInputImage::Pointer &image )
{
  std::string strFileInput( fileInput );

  if ( description )
  {
    std::string strDescription( description );
    return ReadImageFromFile( strFileInput, image, strDescription );
  }
  else 
  {
    return ReadImageFromFile( strFileInput, image );
  }
}


//  --------------------------------------------------------------------------
/// Read an ITK image to a file and print a message
//  --------------------------------------------------------------------------

template < typename TInputImage >
bool
ReadImageFromFile( const char *fileInput, const char *description,
                 typename TInputImage::ConstPointer &image )
{
  std::string strFileInput( fileInput );

  if ( description )
  {
    std::string strDescription( description );
    return ReadImageFromFile( strFileInput, image, strDescription );
  }
  else 
  {
    return ReadImageFromFile( strFileInput, image );
  }
}


//  --------------------------------------------------------------------------
/// Read an ITK image to a file and print a message
//  --------------------------------------------------------------------------

template < typename TInputImage >
bool
ReadImageFromFile( std::string fileInput, typename TInputImage::Pointer &image, 
                   std::string *description=0 )
{
  if ( ( fileInput.length() > 0 ) && niftk::FileExists( fileInput ) ) 
  {

    typedef itk::ImageFileReader< TInputImage > FileReaderType;

    typename FileReaderType::Pointer reader = FileReaderType::New();

    reader->SetFileName( fileInput );

    if ( description )
    {
      std::cout << "Reading " << description << " from file: "
                << fileInput << std::endl;
    }

    reader->Update();
    image = reader->GetOutput();

    return true;
  }

  return false;
}

//  --------------------------------------------------------------------------
/// Read an ITK image to a file and print a message
//  --------------------------------------------------------------------------

template < typename TInputImage >
bool
ReadImageFromFile( std::string fileInput, typename TInputImage::ConstPointer &image, 
                   std::string *description=0 )
{
  if ( ( fileInput.length() > 0 ) && niftk::FileExists( fileInput ) ) 
  {

    typedef itk::ImageFileReader< TInputImage > FileReaderType;

    typename FileReaderType::Pointer reader = FileReaderType::New();

    reader->SetFileName( fileInput );

    if ( description )
    {
      std::cout << "Reading " << description << " from file: "
                << fileInput << std::endl;
    }

    reader->Update();
    image = reader->GetOutput();

    return true;
  }

  return false;
}




} // end namespace itk

#endif /* __itkReadImage_h */
