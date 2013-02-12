/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <fstream>

// ITK
#include "itkImage.h"
#include "itkImageRegionConstIterator.h"
#include "itkMetaDataDictionary.h"
#include "itkMetaDataObject.h"
#include "itkSpatialOrientationAdapter.h"

// MITK
#include "niftiImageToMitk.h"
#include "mitkBaseProcess.h"
#include "mitkImageAccessByItk.h"


// ---------------------------------------------------------------------------
// Normalize()
// ---------------------------------------------------------------------------

void Normalize(std::vector<double> &x)
{
  double sum = 0;

  for (unsigned int i = 0; i < x.size(); i++)
  {
    sum += (x[i] * x[i]);
  }

  if (sum == 0.0)
  {
    return;
  }

  sum = vcl_sqrt(sum);
  for (unsigned int i = 0; i < x.size(); i++)
  {
    x[i] = x[i] / sum;
  }
}

  
// ---------------------------------------------------------------------------
// SetItkOrientationFromNiftiImage()
// ---------------------------------------------------------------------------

template< typename TPixel, unsigned int VImageDimension >
  void SetItkOrientationFromNiftiImage( itk::Image< TPixel, VImageDimension > *itkImage,
					nifti_image *niftiImage )
{
  typedef itk::Image<TPixel, VImageDimension> ImageType;
  typedef itk::SpatialOrientationAdapter OrientAdapterType;

  typename ImageType::PointType origin;


  // In the case of an Analyze75 image, use old analyze orient method.

  if ( niftiImage->qform_code == 0
       && niftiImage->sform_code == 0 )
  {
    typename OrientAdapterType::DirectionType dir;
    typename OrientAdapterType::OrientationType orient;

    switch(niftiImage->analyze75_orient)
    {

    case a75_transverse_unflipped:
      orient = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RPI;
      break;

    case a75_sagittal_unflipped:
      orient = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_PIR;
      break;

      // According to analyze documents, you don't see flipped
      // orientation in the wild
    case a75_transverse_flipped:
    case a75_coronal_flipped:
    case a75_sagittal_flipped:
    case a75_orient_unknown:
    case a75_coronal_unflipped:
      orient = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RIP;
      break;
    }

    dir =  OrientAdapterType().ToDirectionCosines(orient);

    origin[0] = 0;
    origin[1] = 0;

    if (VImageDimension > 2)
    {
      origin[2] = 0;
    }
    return;
  }


  // Not an Analyze image.
  // Scale image data based on slope/intercept
  //
  // qform or sform

  mat44 mat;

  // [Change by Marc next line - Use the sform first and if it's not defined, use the qform then]

  // if(niftiImage->qform_code > 0)
  if (niftiImage->sform_code > 0)
  {
    mat = niftiImage->sto_xyz;
  }

  // else if(niftiImage->sform_code > 0)
  else
  {
    mat = niftiImage->qto_xyz;
  }


  // Set the origin

  origin[0] = -mat.m[0][3];

  if (VImageDimension > 1)
  {
    origin[1] = -mat.m[1][3];
  }

  if (VImageDimension > 2)
  {
    origin[2] = mat.m[2][3];
  }

  itkImage->SetOrigin( origin );


  // Set the orientations

  const int maxDefinedOrientationDims = (VImageDimension > 3) ? 3 : VImageDimension;

  typename ImageType::DirectionType directions;
  directions.Fill( 0 );

  std::vector<double> xDirection(VImageDimension, 0);
  std::vector<double> yDirection(VImageDimension, 0);
  std::vector<double> zDirection(VImageDimension, 0);

  for (int i = 0; i<maxDefinedOrientationDims; i++)
  {
    xDirection[i] = mat.m[i][0];
    if (i < 2)
    {
      xDirection[i] *= -1.0;
    }
  }
  Normalize(xDirection);

  if (maxDefinedOrientationDims > 1 )
  {
    for (int i = 0; i < maxDefinedOrientationDims; i++)
    {
      yDirection[i] = mat.m[i][1];
      if(i < 2)
      {
        yDirection[i] *= -1.0;
      }
    }
    Normalize(yDirection);
  }

  if (maxDefinedOrientationDims > 2 )
  {
    for (int i = 0; i < maxDefinedOrientationDims; i++)
    {
      zDirection[i] = mat.m[i][2];
      if(i < 2)
      {
        zDirection[i] *= -1.0;
      }
    }
    Normalize(zDirection);
  }

  for (int i = 0; i < maxDefinedOrientationDims; i++)
  {
    directions[i][0] = xDirection[i];
    directions[i][1] = yDirection[i];
    directions[i][2] = zDirection[i];
  }

  itkImage->SetDirection( directions );
}


// ---------------------------------------------------------------------------
// ConvertNiftiImageToMitkPixel()
// ---------------------------------------------------------------------------

template< typename TPixel, unsigned int VImageDimension >
mitk::Image::Pointer ConvertNiftiImageToMitkPixel( nifti_image *imageNifti )
{
  unsigned int i;

  typedef itk::Image<TPixel, VImageDimension> ImageType;

  typename ImageType::RegionType myRegion;
  typename ImageType::SizeType mySize;
  typename ImageType::IndexType myIndex;
  typename ImageType::SpacingType mySpacing;

  typename ImageType::Pointer imageITK = ImageType::New();

  // Create new, empty MITK image
  typename mitk::Image::Pointer imageMITK = mitk::Image::New();


  for ( i=1; i<= VImageDimension; i++ ) 
  {
    myIndex[ i - 1 ] = 0;
    mySize[ i - 1 ] = imageNifti->dim[i]; // no. of voxels in dimension i

    mySpacing[ i - 1 ] = imageNifti->pixdim[i];
  }

  myRegion.SetIndex( myIndex );
  myRegion.SetSize( mySize);

  imageITK->SetSpacing(mySpacing);
  imageITK->SetRegions( myRegion);

  imageITK->Allocate();
  imageITK->FillBuffer(0);


  TPixel *pNiftiPixels = static_cast<TPixel*>( imageNifti->data );

  itk::ImageRegionIterator<ImageType>  iterator(imageITK, imageITK->GetLargestPossibleRegion());

  iterator.GoToBegin();

  while (!iterator.IsAtEnd())
  {
    iterator.Set( *pNiftiPixels );
    ++iterator;
    pNiftiPixels++;
  }

  nifti_image_infodump( imageNifti );

  SetItkOrientationFromNiftiImage<TPixel, VImageDimension>( imageITK, imageNifti );

  mitk::CastToMitkImage( imageITK, imageMITK );

  mitk::Geometry3D* geometry = imageMITK->GetGeometry();

  std::ofstream fout("/scratch0/NOT_BACKED_UP/JamiesAffineRegnHeaderTestData/testConvertNiftiImageToMitkPixel_MITKGeometry.txt");
  geometry->Print( fout );
  fout.close();

  return imageMITK;
}
     

// ---------------------------------------------------------------------------
// ConvertNiftiImageToMitkDimension()
// ---------------------------------------------------------------------------

template< unsigned int VImageDimension >
mitk::Image::Pointer ConvertNiftiImageToMitkDimension( nifti_image *imageNifti )
{

  // Create ITK Image + Geometry

  switch ( imageNifti->datatype )
  {
  case DT_UINT8: 
  {
    return ConvertNiftiImageToMitkPixel<unsigned char, VImageDimension>( imageNifti );
    break;
  }
  case DT_INT16: 
  {
    return ConvertNiftiImageToMitkPixel<signed short, VImageDimension>( imageNifti );
    break;
  }
  case DT_UINT16: 
  {
    return ConvertNiftiImageToMitkPixel<unsigned short, VImageDimension>( imageNifti );
    break;
  }
  case DT_INT32: 
  {
    return ConvertNiftiImageToMitkPixel<signed int, VImageDimension>( imageNifti );
    break;
  }
  case DT_UINT32: 
  {
    return ConvertNiftiImageToMitkPixel<unsigned int, VImageDimension>( imageNifti );
    break;
  }
  case DT_FLOAT32: 
  {
    return ConvertNiftiImageToMitkPixel<float, VImageDimension>( imageNifti );
    break;
  }
  case DT_FLOAT64: 
  {
    return ConvertNiftiImageToMitkPixel<double, VImageDimension>( imageNifti );
    break;
  }
  default: 
  {
    MITK_ERROR << "Nifti image type not currently " 
	       << "supported for conversion to MITK" << std::endl;
    return 0;
  }
  }

  return 0;
}


// ---------------------------------------------------------------------------
// ConvertNiftiImageToMitk()
// ---------------------------------------------------------------------------

mitk::Image::Pointer ConvertNiftiImageToMitk( nifti_image *imageNifti )
{

  // Create ITK Image + Geometry

  switch ( imageNifti->dim[0] )
  {
  case 1: 
  {
    return ConvertNiftiImageToMitkDimension< 1 >( imageNifti );
    break;
  }
  case 2: 
  {
    return ConvertNiftiImageToMitkDimension< 2 >( imageNifti );
    break;
  }
  case 3: 
  {
    return ConvertNiftiImageToMitkDimension< 3 >( imageNifti );
    break;
  }
  case 4: 
  {
    return ConvertNiftiImageToMitkDimension< 4 >( imageNifti );
    break;
  }
  case 5: 
  {
    return ConvertNiftiImageToMitkDimension< 5 >( imageNifti );
    break;
  }
  case 6: 
  {
    return ConvertNiftiImageToMitkDimension< 6 >( imageNifti );
    break;
  }
  case 7: 
  {
    return ConvertNiftiImageToMitkDimension< 7 >( imageNifti );
    break;
  }
  default: 
  {
    MITK_ERROR << "Nifti image type not currently " 
	       << "supported for conversion to MITK" << std::endl;
    return 0;
  }
  }

  return 0;
}
       
