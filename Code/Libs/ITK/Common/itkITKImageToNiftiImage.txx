/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <fstream>
#include <iomanip>


// ITK
#include <itkImage.h>
#include <itkImageRegionConstIterator.h>
#include <itkImageFileWriter.h>
#include <itkNiftiImageIO3201.h>

// ITK
#include "itkITKImageToNiftiImage.h"

namespace
{
inline mat44 mat44_transpose(mat44 in)
{
  mat44 out;
  for(unsigned int i = 0; i < 4; i++)
    {
    for(unsigned int j = 0; j < 4; j++)
      {
      out.m[i][j] = in.m[j][i];
      }
    }
  return out;
}
}

// ---------------------------------------------------------------------------
// SetNiftiOrientationFromItkImage()
// ( Lifted from: NiftiImageIO3201 )
// ---------------------------------------------------------------------------

template< typename ITK_VOXEL_TYPE, unsigned int VImageDimension >
void SetNiftiOrientationFromItkImage( nifti_image *niftiImage,
				      typename itk::Image< ITK_VOXEL_TYPE, VImageDimension >::Pointer itkImage )
{

  unsigned short int origdims = VImageDimension;
  unsigned short int dims     = VImageDimension;

  //
  // use NIFTI method 2
  // Scanner-based anatomical coordinates = 1
  niftiImage->sform_code = NIFTI_XFORM_SCANNER_ANAT;
  // Coordinates aligned to another file's, or to anatomical "truth" = 2
  niftiImage->qform_code = NIFTI_XFORM_ALIGNED_ANAT; 

  // set the quarternions, from the direction vectors
  //Initialize to size 3 with values of 0
  //
  //The type here must be float, because that matches the signature
  //of the nifti_make_orthog_mat44() method below.
    //
    // Please note: direction cosines are stored as columns of the
    // direction matrix
  typedef float DirectionMatrixComponentType;
  int mindims(dims < 3 ? 3 : dims);
  std::vector<DirectionMatrixComponentType> dirx(mindims,0);
  unsigned int i;
  for(i=0; i < (unsigned int) itkImage->GetDirection().ColumnDimensions; i++)
  {
    dirx[i] = static_cast<DirectionMatrixComponentType>(-itkImage->GetDirection()[i][0]);
  }
  if(i < 3)
  {
    dirx[2] = 0.0f;
  }
  std::vector<DirectionMatrixComponentType> diry(mindims,0);
  if(origdims > 1)
  {
    for(i=0; i < (unsigned int) itkImage->GetDirection().ColumnDimensions; i++)
    {
      diry[i] = static_cast<DirectionMatrixComponentType>(-itkImage->GetDirection()[i][1]);
    }
    if(i < 3)
    {
      diry[2] = 0.0f;
    }
  }
  std::vector<DirectionMatrixComponentType> dirz(mindims,0);
  if(origdims > 2)
  {
    for(unsigned int ii=0; ii < (unsigned int) itkImage->GetDirection().ColumnDimensions; ii++)
    {
      dirz[ii] = static_cast<DirectionMatrixComponentType>( -itkImage->GetDirection()[ii][2] );
    }
    //  Read comments in nifti1.h about interpreting
    //  "DICOM Image Orientation (Patient)"
    dirx[2] = - dirx[2];
    diry[2] = - diry[2];
    dirz[2] = - dirz[2];
  }
  else
  {
    dirz[0] = dirz[1] = 0.0f;
    dirz[2] = 1.0f;
  }

  mat44 matrix =
    nifti_make_orthog_mat44(dirx[0],dirx[1],dirx[2],
			    diry[0],diry[1],diry[2],
			    dirz[0],dirz[1],dirz[2]);

  matrix = mat44_transpose(matrix);
  // Fill in origin.
  matrix.m[0][3]=  static_cast<float>(-itkImage->GetOrigin()[0]);
  matrix.m[1][3] = (origdims > 1) ? static_cast<float>(-itkImage->GetOrigin()[1]) : 0.0f;
  //NOTE:  The final dimension is not negated!
  matrix.m[2][3] = (origdims > 2) ? static_cast<float>(itkImage->GetOrigin()[2]) : 0.0f;

  nifti_mat44_to_quatern(matrix,
			 &(niftiImage->quatern_b),
			 &(niftiImage->quatern_c),
			 &(niftiImage->quatern_d),
			 &(niftiImage->qoffset_x),
			 &(niftiImage->qoffset_y),
			 &(niftiImage->qoffset_z),
			 0,
			 0,
			 0,
			 &(niftiImage->qfac));

  // copy q matrix to s matrix
  niftiImage->qto_xyz =  matrix;
  niftiImage->sto_xyz =  matrix;

  //
  //

  unsigned int sto_limit = origdims > 3 ? 3 : origdims;

  for(unsigned int ii = 0; ii < sto_limit; ii++)
  {
    for(unsigned int jj = 0; jj < sto_limit; jj++)
    {

      niftiImage->sto_xyz.m[ii][jj] =
	static_cast<float>( itkImage->GetSpacing()[jj] ) * niftiImage->sto_xyz.m[ii][jj];

      niftiImage->qto_xyz.m[ii][jj] =
	static_cast<float>( itkImage->GetSpacing()[jj] ) * niftiImage->qto_xyz.m[ii][jj];
    }
  }

  niftiImage->sto_ijk =
    nifti_mat44_inverse(niftiImage->sto_xyz);
  niftiImage->qto_ijk =
    nifti_mat44_inverse(niftiImage->qto_xyz);

  niftiImage->pixdim[0] = niftiImage->qfac;
  //  niftiImage->sform_code = 0;
}


// ---------------------------------------------------------------------------
// CopyItkIntensitiesToNifti()
// ---------------------------------------------------------------------------

template<typename ITK_VOXEL_TYPE, typename NIFTI_VOXEL_TYPE, unsigned int VImageDimension>
void CopyItkIntensitiesToNifti(itk::Image<ITK_VOXEL_TYPE, VImageDimension>* itkImage, 
				   nifti_image *niftiImage)
{
  typedef typename itk::Image<ITK_VOXEL_TYPE, VImageDimension> ItkImageType;

  itk::ImageRegionIterator<ItkImageType> inputIterator( itkImage, 
							itkImage->GetLargestPossibleRegion() );

  NIFTI_VOXEL_TYPE *pNiftiPixels = static_cast<NIFTI_VOXEL_TYPE*>( niftiImage->data );

  for ( inputIterator.GoToBegin(); ! inputIterator.IsAtEnd(); ++inputIterator )
  {
    *pNiftiPixels = inputIterator.Get();
    pNiftiPixels++;
  }

}


// ---------------------------------------------------------------------------
// ConvertITKImageToNiftiImage()
// ---------------------------------------------------------------------------

template<typename NIFTI_PRECISION_TYPE, typename ITK_VOXEL_TYPE, unsigned int VImageDimension>
nifti_image *ConvertITKImageToNiftiImage( typename itk::Image< ITK_VOXEL_TYPE, VImageDimension >::Pointer itkImage )
{
  int i, nBytesPerVoxel, swapsize;

  typedef itk::Image< ITK_VOXEL_TYPE, VImageDimension > ImageType;

  typename ImageType::SizeType size;

  size = itkImage->GetLargestPossibleRegion().GetSize();

  typename ImageType::SpacingType spacing;

  spacing = itkImage->GetSpacing();


  // Create the nifti header
  // ~~~~~~~~~~~~~~~~~~~~~~~

  struct nifti_1_header niftiHeader;

  // zero out header, to be safe
  memset( &niftiHeader, 0, sizeof(niftiHeader) );
  
  niftiHeader.sizeof_hdr = sizeof(niftiHeader);

  niftiHeader.regular = 'r';

  // The image dimension
  niftiHeader.dim[0] = VImageDimension; 

  // Set the number of voxels
  for ( i=1; i<= niftiHeader.dim[0]; i++ )
    niftiHeader.dim[i] = size[ i - 1 ]; // no. of voxels in dimension i

  // We assume the ITK convention of units of millmeters and seconds
  //niftiHeader.xyzt_units = NIFTI_UNITS_MM | NIFTI_UNITS_SEC;
  niftiHeader.xyzt_units = NIFTI_UNITS_MM;

  // Set the voxel resolution
  niftiHeader.pixdim[0] = 0.0;	// Undefined

  for ( i=1; i<= niftiHeader.dim[0]; i++ ) 
    niftiHeader.pixdim[i] = spacing[ i - 1 ];

  // Set the data type
  if ( typeid(NIFTI_PRECISION_TYPE) == typeid(unsigned char) )
  {
    niftiHeader.datatype = DT_UINT8;
  } 
  else if ( typeid(NIFTI_PRECISION_TYPE) == typeid(signed char) )
  {
    niftiHeader.datatype = DT_INT8;
  } 
  else if ( typeid(NIFTI_PRECISION_TYPE) == typeid(signed short) )
  {
    niftiHeader.datatype = DT_INT16;
  } 
  else if ( typeid(NIFTI_PRECISION_TYPE) == typeid(unsigned short) ) 
  {
    niftiHeader.datatype = DT_UINT16;
  }
  else if ( typeid(NIFTI_PRECISION_TYPE) == typeid(signed int) )
  {
    niftiHeader.datatype = DT_INT32;
  } 
  else if ( typeid(NIFTI_PRECISION_TYPE) == typeid(unsigned int) ) 
  {
    niftiHeader.datatype = DT_UINT32;
  }
  else if ( typeid(NIFTI_PRECISION_TYPE) == typeid(float) ) 
  {
    niftiHeader.datatype = DT_FLOAT32;
  } 
  else if ( typeid(NIFTI_PRECISION_TYPE) == typeid(double) ) 
  {
    niftiHeader.datatype = DT_FLOAT64;
  }
  else 
  {
    std::cerr << "ERROR: Nifti precision type not currently " 
              << "supported for conversion from ITK" << std::endl;
    return 0;
  }

  nifti_datatype_sizes( niftiHeader.datatype, &nBytesPerVoxel, &swapsize );

  niftiHeader.bitpix = 8*nBytesPerVoxel; // Number bits/voxel

  // init to single file
  strcpy( niftiHeader.magic, "n+1" ); 	// NIFTI-1 flag

  if ( niftiHeader.dim[1] < 1 ) niftiHeader.dim[1] = 1;
  if ( niftiHeader.dim[2] < 1 ) niftiHeader.dim[2] = 1;
  if ( niftiHeader.dim[3] < 1 ) niftiHeader.dim[3] = 1;
  if ( niftiHeader.dim[4] < 1 ) niftiHeader.dim[4] = 1;
  if ( niftiHeader.dim[5] < 1 ) niftiHeader.dim[5] = 1;
  if ( niftiHeader.dim[6] < 1 ) niftiHeader.dim[6] = 1;
  if ( niftiHeader.dim[7] < 1 ) niftiHeader.dim[7] = 1;
				                        
  if ( niftiHeader.scl_slope == 0 ) niftiHeader.scl_slope = 1.f;


  // Allocate the nifti image
  // ~~~~~~~~~~~~~~~~~~~~~~~~

  nifti_image *niftiImage = 0;
  
  niftiImage = nifti_convert_nhdr2nim( niftiHeader, NULL );
  niftiImage->data = calloc( niftiImage->nvox, niftiImage->nbyper );

  niftiImage->fname = NULL;
  niftiImage->iname = NULL;


  // Copy the voxel intensity data across
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  typedef typename itk::Image<ITK_VOXEL_TYPE, VImageDimension> ImageType;
  
  switch (niftiImage->datatype)
  {
    case DT_UINT8:
    {
      CopyItkIntensitiesToNifti<ITK_VOXEL_TYPE, unsigned char, VImageDimension>(itkImage, niftiImage);
      break;
    }

    case DT_INT8:
    {
      CopyItkIntensitiesToNifti<ITK_VOXEL_TYPE, char, VImageDimension>(itkImage, niftiImage);
      break;
    }

    case DT_INT16:
    {
      CopyItkIntensitiesToNifti<ITK_VOXEL_TYPE, short, VImageDimension>(itkImage, niftiImage);
      break;
    }

    case DT_UINT16:
    {
      CopyItkIntensitiesToNifti<ITK_VOXEL_TYPE, unsigned short, VImageDimension>(itkImage, niftiImage);
      break;
    }

    case DT_INT32:
    {
      CopyItkIntensitiesToNifti<ITK_VOXEL_TYPE, int, VImageDimension>(itkImage, niftiImage);
      break;
    }

    case DT_UINT32:
    {
      CopyItkIntensitiesToNifti<ITK_VOXEL_TYPE, unsigned int, VImageDimension>(itkImage, niftiImage);
      break;
    }

    case DT_FLOAT32:
    {
      std::cout << "Copying FLOAT data" << std::endl;

      CopyItkIntensitiesToNifti<ITK_VOXEL_TYPE, float, VImageDimension>(itkImage, niftiImage);
      break;
    }

    case DT_FLOAT64:
    {
      CopyItkIntensitiesToNifti<ITK_VOXEL_TYPE, double, VImageDimension>(itkImage, niftiImage);
      break;
    }

    default:
    {
      std::cerr << "ERROR: Nifti precision type not currently " 
                << "supported for conversion from MITK" << std::endl;
      return 0;
    }
  }


  SetNiftiOrientationFromItkImage<ITK_VOXEL_TYPE, VImageDimension>( niftiImage, itkImage );



  return niftiImage;
}
