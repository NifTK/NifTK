/**************************************************************************
 * NifTK Note: This is take from ITK 3.20.1 (hence the name), which was
 * the version of ITK used within NifTK at the time of writing. The
 * version 3.20.1 actually signified a patched version of ITK, and
 * the patched tar file and patch can be seen in:
 *   NifTK/CMake/ExternalProjects/ITK.cmake and
 *   NifTK/CMake/ExternalProjects/PatchITK-3.20.cmake
 * However, none of this patching affects the Nifti reader, so the base
 * version of the Nifti reader used for this class is the same as in 3.20.0.
 *
 * The changes supplied here are to include the sform in the transformation.
 **************************************************************************/

/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    itkNiftiImageIO.cxx
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/

#include "itkNiftiImageIO3201.h"
#include <itkIOCommon.h>
#include <itkExceptionObject.h>
#include <itkByteSwapper.h>
#include <itkMetaDataObject.h>
#include <itkSpatialOrientationAdapter.h>
#include <itkNumericTraits.h>
#include <itksys/SystemTools.hxx>
#include <vnl/vnl_math.h>
#include <itk_zlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

namespace niftk
{
//#define __USE_VERY_VERBOSE_NIFTI_DEBUGGING__
#if defined(__USE_VERY_VERBOSE_NIFTI_DEBUGGING__)
namespace
{
static int print_hex_vals(
  char const * const data,
  const int nbytes,
  FILE * const fp )
{
  int c;

  if ( !data || nbytes < 1 || !fp )
    {
    return -1;
    }
  fputs("0x", fp);
  for ( c = 0; c < nbytes; c++ )
    {
    fprintf(fp, " %x", data[c]);
    }

  return 0;
}
static char *str_intent(unsigned int intent)
{
  switch(intent)
    {
    case NIFTI_INTENT_NONE:
      return "NIFTI_INTENT_NONE";
    case NIFTI_INTENT_CORREL:
      return "NIFTI_INTENT_CORREL";
    case NIFTI_INTENT_TTEST:
      return "NIFTI_INTENT_TTEST";
    case NIFTI_INTENT_FTEST:
      return "NIFTI_INTENT_FTEST";
    case NIFTI_INTENT_ZSCORE:
      return "NIFTI_INTENT_ZSCORE";
    case NIFTI_INTENT_CHISQ:
      return "NIFTI_INTENT_CHISQ";
    case NIFTI_INTENT_BETA:
      return "NIFTI_INTENT_BETA";
    case NIFTI_INTENT_BINOM:
      return "NIFTI_INTENT_BINOM";
    case NIFTI_INTENT_GAMMA:
      return "NIFTI_INTENT_GAMMA";
    case NIFTI_INTENT_POISSON:
      return "NIFTI_INTENT_POISSON";
    case NIFTI_INTENT_NORMAL:
      return "NIFTI_INTENT_NORMAL";
    case NIFTI_INTENT_FTEST_NONC:
      return "NIFTI_INTENT_FTEST_NONC";
    case NIFTI_INTENT_CHISQ_NONC:
      return "NIFTI_INTENT_CHISQ_NONC";
    case NIFTI_INTENT_LOGISTIC:
      return "NIFTI_INTENT_LOGISTIC";
    case NIFTI_INTENT_LAPLACE:
      return "NIFTI_INTENT_LAPLACE";
    case NIFTI_INTENT_UNIFORM:
      return "NIFTI_INTENT_UNIFORM";
    case NIFTI_INTENT_TTEST_NONC:
      return "NIFTI_INTENT_TTEST_NONC";
    case NIFTI_INTENT_WEIBULL:
      return "NIFTI_INTENT_WEIBULL";
    case NIFTI_INTENT_CHI:
      return "NIFTI_INTENT_CHI";
    case NIFTI_INTENT_INVGAUSS:
      return "NIFTI_INTENT_INVGAUSS";
    case NIFTI_INTENT_EXTVAL:
      return "NIFTI_INTENT_EXTVAL";
    case NIFTI_INTENT_PVAL:
      return "NIFTI_INTENT_PVAL";
    case NIFTI_INTENT_LOGPVAL:
      return "NIFTI_INTENT_LOGPVAL";
    case NIFTI_INTENT_LOG10PVAL:
      return "NIFTI_INTENT_LOG10PVAL";
    case NIFTI_INTENT_ESTIMATE:
      return "NIFTI_INTENT_ESTIMATE";
    case NIFTI_INTENT_LABEL:
      return "NIFTI_INTENT_LABEL";
    case NIFTI_INTENT_NEURONAME:
      return "NIFTI_INTENT_NEURONAME";
    case NIFTI_INTENT_GENMATRIX:
      return "NIFTI_INTENT_GENMATRIX";
    case NIFTI_INTENT_SYMMATRIX:
      return "NIFTI_INTENT_SYMMATRIX";
    case NIFTI_INTENT_DISPVECT:
      return "NIFTI_INTENT_DISPVECT";
    case NIFTI_INTENT_VECTOR:
      return "NIFTI_INTENT_VECTOR";
    case NIFTI_INTENT_POINTSET:
      return "NIFTI_INTENT_POINTSET";
    case NIFTI_INTENT_TRIANGLE:
      return "NIFTI_INTENT_TRIANGLE";
    case NIFTI_INTENT_QUATERNION:
      return "NIFTI_INTENT_QUATERNION";
    case NIFTI_INTENT_DIMLESS:
      return "NIFTI_INTENT_DIMLESS";
    default:
      return "UNKNOWN_INTENT";
    }
}
/*----------------------------------------------------------------------*/
/*! display the contents of the nifti_1_header (send to stdout)
 *--------------------------------------------------------------------*/
static int DumpNiftiHeader( const std::string &fname )
{
  int c;
  nifti_1_header *hp;
  int swap;
  hp = nifti_read_header(fname.c_str(),&swap,true);
  fputs( "-------------------------------------------------------\n",
    stderr );
  if ( !hp  )
    {
    fputs(" ** no nifti_1_header to display!\n",stderr);
    return 1;
    }

  fprintf(stderr," nifti_1_header :\n"
    "    sizeof_hdr     = %d\n"
    "    data_type[10]  = ", hp->sizeof_hdr);
  print_hex_vals(hp->data_type, 10, stderr);
  fprintf(stderr, "\n"
    "    db_name[18]    = ");
  print_hex_vals(hp->db_name, 18, stderr);
  fprintf(stderr, "\n"
    "    extents        = %d\n"
    "    session_error  = %d\n"
    "    regular        = 0x%x\n"
    "    dim_info       = 0x%x\n",
    hp->extents, hp->session_error, hp->regular, hp->dim_info );
  fprintf(stderr, "    dim[8]         =");
  for ( c = 0; c < 8; c++ ) fprintf(stderr," %d", hp->dim[c]);
  fprintf(stderr, "\n"
    "    intent_p1      = %f\n"
    "    intent_p2      = %f\n"
    "    intent_p3      = %f\n"
    "    intent_code    = %s\n"
    "    datatype       = %d\n"
    "    bitpix         = %d\n"
    "    slice_start    = %d\n"
    "    pixdim[8]      =",
    hp->intent_p1, hp->intent_p2, hp->intent_p3,
    str_intent(hp->intent_code),
    hp->datatype, hp->bitpix, hp->slice_start);
  /* break pixdim over 2 lines */
  for ( c = 0; c < 4; c++ ) fprintf(stderr," %f", hp->pixdim[c]);
  fprintf(stderr, "\n                    ");
  for ( c = 4; c < 8; c++ ) fprintf(stderr," %f", hp->pixdim[c]);
  fprintf(stderr, "\n"
    "    vox_offset     = %f\n"
    "    scl_slope      = %f\n"
    "    scl_inter      = %f\n"
    "    slice_end      = %d\n"
    "    slice_code     = %d\n"
    "    xyzt_units     = 0x%x\n"
    "    cal_max        = %f\n"
    "    cal_min        = %f\n"
    "    slice_duration = %f\n"
    "    toffset        = %f\n"
    "    glmax          = %d\n"
    "    glmin          = %d\n",
    hp->vox_offset, hp->scl_slope, hp->scl_inter, hp->slice_end,
    hp->slice_code, hp->xyzt_units, hp->cal_max, hp->cal_min,
    hp->slice_duration, hp->toffset, hp->glmax, hp->glmin);
  fprintf(stderr,
    "    descrip        = '%.80s'\n"
    "    aux_file       = '%.24s'\n"
    "    qform_code     = %d\n"
    "    sform_code     = %d\n"
    "    quatern_b      = %f\n"
    "    quatern_c      = %f\n"
    "    quatern_d      = %f\n"
    "    qoffset_x      = %f\n"
    "    qoffset_y      = %f\n"
    "    qoffset_z      = %f\n"
    "    srow_x[4]      = %f, %f, %f, %f\n"
    "    srow_y[4]      = %f, %f, %f, %f\n"
    "    srow_z[4]      = %f, %f, %f, %f\n"
    "    intent_name    = '%-.16s'\n"
    "    magic          = '%-.4s'\n",
    hp->descrip, hp->aux_file, hp->qform_code, hp->sform_code,
    hp->quatern_b, hp->quatern_c, hp->quatern_d,
    hp->qoffset_x, hp->qoffset_y, hp->qoffset_z,
    hp->srow_x[0], hp->srow_x[1], hp->srow_x[2], hp->srow_x[3],
    hp->srow_y[0], hp->srow_y[1], hp->srow_y[2], hp->srow_y[3],
    hp->srow_z[0], hp->srow_z[1], hp->srow_z[2], hp->srow_z[3],
    hp->intent_name, hp->magic);
  fputs("-------------------------------------------------------\n",
    stderr );
  fflush(stderr);

  return 0;
}
static void dumpdata(const void *x)
{
  std::cerr << "----------------------" << std::endl;

  //    typedef const float (*itkarray)[1][2][2][2][3];
  const float *a = (const float *)x;
  for(unsigned int i = 0; i < 24; i++)         // t
    {
    std::cerr << a[i] << std::endl;
    }
}
}
#else
#define dumpdata(x)
#endif // #if defined(__USE_VERY_VERBOSE_NIFTI_DEBUGGING__)

// returns an ordering array for converting upper triangular symmetric matrix
// to lower triangular symmetric matrix
int *
UpperToLowerOrder(int dim)
{
  int **mat = new int *[dim];
  for(int i = 0; i < dim; i++)
    {
    mat[i] = new int[dim];
    }
  // fill in
  int index(0);
  for(int i = 0; i < dim; i++)
    {
    for(int j = i; j < dim; j++)
      {
      mat[i][j] = index;
      mat[j][i] = index;
      index++;
      }
    }
  int *rval = new int[index+1];
  int index2(0);
  for(int i = 0; i < dim; i++)
    {
    for(int j = 0; j <= i; j++,index2++)
      {
      rval[index2] = mat[i][j];
      }
    }
  rval[index2] = -1;
  for(int i = 0; i < dim; i++)
    {
    delete [] mat[i];
    }
  delete [] mat;
  return rval;
}
// returns an ordering array for converting lower triangular symmetric matrix
// to upper triangular symmetric matrix
int *
LowerToUpperOrder(int dim)
{
  int **mat = new int *[dim];
  for(int i = 0; i < dim; i++)
    {
    mat[i] = new int[dim];
    }
  // fill in
  int index(0);
  for(int i = 0; i < dim; i++)
    {
    for(int j = 0; j <= i; j++,index++)
      {
      mat[i][j] = index;
      mat[j][i] = index;
      }
    }
  int *rval = new int[index+1];
  int index2(0);
  for(int i = 0; i < dim; i++)
    {
    for(int j = i; j < dim; j++,index2++)
      {
      rval[index2] = mat[i][j];
      }
    }
  rval[index2] = -1;
  for(int i = 0; i < dim; i++)
    {
    delete [] mat[i];
    }
  delete [] mat;
  return rval;
}
// compute the rank of the symmetric matrix from
// the count of the triangular matrix elements
int SymMatDim(int count)
{
  int dim = 0;
  int row = 1;
  while(count > 0)
    {
    count -= row;
    dim++;
    row++;
    }
  return dim;
}

} // end namespace niftk

namespace itk
{

ImageIORegion
NiftiImageIO3201
::GenerateStreamableReadRegionFromRequestedRegion(const ImageIORegion & requestedRegion ) const
{
  return requestedRegion;
}


NiftiImageIO3201::NiftiImageIO3201():
  m_NiftiImage(0),
  m_RescaleSlope(1.0),
  m_RescaleIntercept(0.0),
  m_OnDiskComponentType(UNKNOWNCOMPONENTTYPE),
  m_LegacyAnalyze75Mode(true)
{
  this->SetNumberOfDimensions(3);
  nifti_set_debug_level(0); // suppress error messages
  this->AddSupportedWriteExtension(".nia");
  this->AddSupportedWriteExtension(".nii");
  this->AddSupportedWriteExtension(".nii.gz");

  this->AddSupportedReadExtension(".nia");
  this->AddSupportedReadExtension(".nii");
  this->AddSupportedReadExtension(".nii.gz");

  this->AddSupportedWriteExtension(".hdr");
  this->AddSupportedWriteExtension(".img");
  this->AddSupportedWriteExtension(".img.gz");
  this->AddSupportedReadExtension(".hdr");
  this->AddSupportedReadExtension(".img");
  this->AddSupportedReadExtension(".img.gz");
}

NiftiImageIO3201::~NiftiImageIO3201()
{
  nifti_image_free(m_NiftiImage);
}

void
NiftiImageIO3201
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
  os << indent << "LegacyAnalyze75Mode: " << m_LegacyAnalyze75Mode << std::endl;
}

bool
NiftiImageIO3201
::CanWriteFile(const char * FileNameToWrite)
{
  const int ValidFileNameFound=nifti_is_complete_filename(FileNameToWrite) > 0;
  return ValidFileNameFound;
}

bool
NiftiImageIO3201::MustRescale()
{
  return vcl_abs(m_RescaleSlope) > vcl_numeric_limits<double>::epsilon() &&
    (vcl_abs(m_RescaleSlope-1.0) > vcl_numeric_limits<double>::epsilon() ||
     vcl_abs(m_RescaleIntercept) > vcl_numeric_limits<double>::epsilon());
}

// Internal function to rescale pixel according to Rescale Slope/Intercept
template<class TBuffer>
void RescaleFunction(TBuffer* buffer,
                     double slope,
                     double intercept,
                     size_t size)
{
  for(unsigned int i=0; i<size; i++)
    {
    double tmp = static_cast<double>(buffer[i]) * slope;
    tmp += intercept;
    buffer[i] = static_cast<TBuffer>(tmp);
    }
}

template <typename PixelType>
void
CastCopy(float *to,void *from, size_t pixelcount)
{
  PixelType *_from = static_cast<PixelType *>(from);
  for(unsigned i = 0; i < pixelcount; i++)
    {
    to[i] = static_cast<float>(_from[i]);
    }
}

void NiftiImageIO3201::Read(void* buffer)
{
  void *data = 0;

  ImageIORegion regionToRead = this->GetIORegion();
  ImageIORegion::SizeType size = regionToRead.GetSize();
  ImageIORegion::IndexType start = regionToRead.GetIndex();

  int numElts = 1;
  int _origin[7];
  int _size[7];
  unsigned int i;
  for(i = 0; i < start.size(); i++)
    {
    _origin[i] = static_cast<int>( start[i] );
    _size[i] = static_cast<int>( size[i] );
    numElts *= _size[i];
    }
  for(; i < 7; i++)
    {
    _origin[i] = 0;
    _size[i] = 1;
    }

  unsigned int numComponents = this->GetNumberOfComponents();
  //
  // special case for images of vector pixels
  if(numComponents > 1 && this->GetPixelType() != COMPLEX)
    {
    // nifti always sticks vec size in dim 4, so have to shove
    // other dims out of the way
    _size[6] = _size[5];
    _size[5] = _size[4];
    // sizes = x y z t vecsize
    _size[4] = numComponents;
    }
  // Free memory if any was occupied already (incase of re-using the IO filter).
  if (m_NiftiImage != NULL)
    {
      nifti_image_free(m_NiftiImage);
    }
  //
  // allocate nifti image...
  m_NiftiImage = nifti_image_read(this->GetFileName(),false);
  if (m_NiftiImage == NULL)
    {
    itkExceptionMacro(<< "nifti_image_read (just header) failed for file: "
                      << this->GetFileName());
    }

  //
  // decide whether to read whole region or subregion, by stepping
  // thru dims and comparing them to requested sizes
  for(i = 0; i < this->GetNumberOfDimensions(); i++)
    {
    if(m_NiftiImage->dim[i+1] != _size[i])
      {
      break;
      }
    }
  // if all dimensions match requested size, just read in
  // all data as a block
  if(i == this->GetNumberOfDimensions())
    {
    if(nifti_image_load(m_NiftiImage) == -1)
      {
      itkExceptionMacro(<< "nifti_image_load failed for file: "
                        << this->GetFileName());
      }
    data = m_NiftiImage->data;
    }
  else
    {
    // read in a subregion
    if(nifti_read_subregion_image(m_NiftiImage,
                                  _origin,
                                  _size,
                                  &data) == -1 || m_NiftiImage == NULL)
      {
      itkExceptionMacro(<< "nifti_read_subregion_image failed for file: "
                        << this->GetFileName());
      }
    }
  unsigned int pixelSize = m_NiftiImage->nbyper;
  //
  // if we're going to have to rescale pixels, and the on-disk
  // pixel type is different than the pixel type reported to
  // ImageFileReader, we have to up-promote the data to float
  // before doing the rescale.
  //
  if(this->MustRescale() &&
     m_ComponentType != m_OnDiskComponentType)
    {
    pixelSize =
      static_cast< unsigned int >( this->GetNumberOfComponents() ) * 
      static_cast< unsigned int >( sizeof(float) );

    // Deal with correct management of 64bits platforms
    const size_t imageSizeInComponents = 
      static_cast< size_t >( this->GetImageSizeInComponents() );

    //
    // allocate new buffer for floats. Malloc instead of new to
    // be consistent with allocation used in niftilib
    float *_data = 
      static_cast<float *>
      (malloc( imageSizeInComponents * sizeof(float)));
    switch(m_OnDiskComponentType)
      {
      case CHAR:
        CastCopy<char>(_data,data, imageSizeInComponents);
        break;
      case UCHAR:
        CastCopy<unsigned char>(_data,data, imageSizeInComponents);
        break;
      case SHORT:
        CastCopy<short>(_data,data, imageSizeInComponents);
        break;
      case USHORT:
        CastCopy<unsigned short>(_data,data, imageSizeInComponents);
        break;
      case INT:
        CastCopy<int>(_data,data, imageSizeInComponents);
        break;
      case UINT:
        CastCopy<unsigned int>(_data,data, imageSizeInComponents);
        break;
      case LONG:
        CastCopy<long>(_data,data, imageSizeInComponents);
        break;
      case ULONG:
        CastCopy<unsigned long>(_data,data, imageSizeInComponents);
        break;
      case FLOAT:
        itkExceptionMacro(<< "FLOAT pixels do not need Casting to float");
        break;
      case DOUBLE:
        itkExceptionMacro(<< "DOUBLE pixels do not need Casting to float");
        break;
      case UNKNOWNCOMPONENTTYPE:
        itkExceptionMacro(<< "Bad OnDiskComponentType UNKNOWNCOMPONENTTYPE");
      }
      //
      // we're replacing the data pointer, so if it was allocated
      // in nifti_read_subregion_image, free the old data here
      if(data != m_NiftiImage->data)
        {
        free(data);
        }
      data = _data;
    }
  //
  // if single or complex, nifti layout == itk layout
  if(numComponents == 1 || 
     this->GetPixelType() == COMPLEX || 
     this->GetPixelType() == RGB || 
     this->GetPixelType() == RGBA)
    {
    const size_t NumBytes= numElts * pixelSize;
    memcpy(buffer, data, NumBytes);
    //
    // if read_subregion was called it allocates a buffer that needs to be
    // freed.
    if(data != m_NiftiImage->data)
      {
      free(data);
      }
    }
  else
    {
    // otherwise nifti is x y z t vec l m 0, itk is
    // vec x y z t l m o
    const char *niftibuf = (const char *)data;
    char *itkbuf = (char *)buffer;
    const unsigned int rowdist=m_NiftiImage->dim[1];
    const unsigned int slicedist=rowdist*m_NiftiImage->dim[2];
    const unsigned int volumedist=slicedist*m_NiftiImage->dim[3];
    const unsigned int seriesdist=volumedist*m_NiftiImage->dim[4];
    //
    // as per ITK bug 0007485
    // NIfTI is lower triangular, ITK is upper triangular.
    int *vecOrder;
    if(this->GetPixelType() == ImageIOBase::DIFFUSIONTENSOR3D ||
       this->GetPixelType() == ImageIOBase::SYMMETRICSECONDRANKTENSOR)
      {
//      vecOrder = LowerToUpperOrder(SymMatDim(numComponents));
        vecOrder = niftk::UpperToLowerOrder(niftk::SymMatDim(numComponents));
      }
    else
      {
      vecOrder = new int[numComponents];
      for(i = 0; i < numComponents; i++)
        {
        vecOrder[i] = i;
        }
      }
    for(int t = 0; t < m_NiftiImage->dim[4]; t++)
      {
      for(int z = 0; z < m_NiftiImage->dim[3]; z++)
        {
        for(int y = 0; y < m_NiftiImage->dim[2]; y++)
          {
          for(int x = 0; x < m_NiftiImage->dim[1]; x++)
            {
            for(unsigned int c=0;c< numComponents; c++)
              {
              const unsigned int nifti_index=(c*seriesdist+volumedist*t + slicedist*z + rowdist*y + x)*pixelSize;
              const unsigned int itk_index=((volumedist*t + slicedist*z + rowdist*y + x)*numComponents + vecOrder[c])*pixelSize;
              memcpy(itkbuf+itk_index,niftibuf+nifti_index,pixelSize);
              }
            }
          }
        }
      }
    delete [] vecOrder;
    dumpdata(data);
    dumpdata(buffer);
    // if read_subregion was called it allocates a buffer that needs to be
    // freed.
    if(data != m_NiftiImage->data)
      {
      free(data);
      }
    }

  // If the scl_slope field is nonzero, then rescale each voxel value in the
  // dataset.
  // Complete description of can be found in nifti1.h under "DATA SCALING"
  if(this->MustRescale())
    {
    switch(m_ComponentType)
      {
      case CHAR:
        RescaleFunction(static_cast<char *>(buffer),
                        m_RescaleSlope,
                        m_RescaleIntercept,numElts);
        break;
      case UCHAR:
        RescaleFunction(static_cast<unsigned char *>(buffer),
                        m_RescaleSlope,
                        m_RescaleIntercept,numElts);
        break;
      case SHORT:
        RescaleFunction(static_cast<short *>(buffer),
                        m_RescaleSlope,
                        m_RescaleIntercept,numElts);
        break;
      case USHORT:
        RescaleFunction(static_cast<unsigned short *>(buffer),
                        m_RescaleSlope,
                        m_RescaleIntercept,numElts);
        break;
      case INT:
        RescaleFunction(static_cast<int *>(buffer),
                        m_RescaleSlope,
                        m_RescaleIntercept,numElts);
        break;
      case UINT:
        RescaleFunction(static_cast<unsigned int *>(buffer),
                        m_RescaleSlope,
                        m_RescaleIntercept,numElts);
        break;
      case LONG:
        RescaleFunction(static_cast<long *>(buffer),
                        m_RescaleSlope,
                        m_RescaleIntercept,numElts);
        break;
      case ULONG:
        RescaleFunction(static_cast<unsigned long *>(buffer),
                        m_RescaleSlope,
                        m_RescaleIntercept,numElts);
        break;
      case FLOAT:
        RescaleFunction(static_cast<float *>(buffer),
                        m_RescaleSlope,
                        m_RescaleIntercept,numElts);
        break;
      case DOUBLE:
        RescaleFunction(static_cast<double *>(buffer),
                        m_RescaleSlope,
                        m_RescaleIntercept,numElts);
        break;
      default:
        if(this->GetPixelType() == SCALAR)
          {
          itkExceptionMacro(<< "Datatype: "
                            << this->GetComponentTypeAsString(m_ComponentType)
                            << " not supported");
          }
      }
    }
}


// This method will only test if the header looks like an
// Nifti Header.  Some code is redundant with ReadImageInformation
// a StateMachine could provide a better implementation
bool
NiftiImageIO3201
::CanReadFile( const char* FileNameToRead )
{
    // is_nifti_file returns
    //       > 0 for a nifti file
    //      == 0 for an analyze file,
    //       < 0 for an error,
    // if the return test is >= 0, nifti will read analyze files
    //return is_nifti_file(FileNameToRead) > 0;
    const int image_FTYPE=is_nifti_file(FileNameToRead);
  if(image_FTYPE>0)
    {
    return true;
    }
  else if (image_FTYPE == 0 && ( this->GetLegacyAnalyze75Mode() == true ))
    {
    return true;
    }
  /* image_FTYPE < 0 */
  return false;
}

// This method adds the available header information to the
// metadata dictionary.
void NiftiImageIO3201::SetImageIOMetadataFromNIfTI()
{
  int swap = 0;
  nifti_1_header *header = nifti_read_header(this->GetFileName(), &swap, true);
  if (header)
    {
    // Encapsulate as many header information as possible.
    MetaDataDictionary &thisDic = this->GetMetaDataDictionary();

    std::ostringstream dim_info;
    dim_info << header->dim_info;
    EncapsulateMetaData<std::string>(thisDic, "dim_info", dim_info.str());

    for (int idx = 0; idx < 8; idx++)
      {
      std::ostringstream dim;
      dim << header->dim[idx];
      std::ostringstream dimKey;
      dimKey << "dim[" << idx << "]";
      EncapsulateMetaData<std::string>(thisDic, dimKey.str(), dim.str());
      }

    std::ostringstream intent_p1;
    intent_p1 << header->intent_p1;
    EncapsulateMetaData<std::string>(thisDic, "intent_p1", intent_p1.str());

    std::ostringstream intent_p2;
    intent_p2 << header->intent_p2;
    EncapsulateMetaData<std::string>(thisDic, "intent_p2", intent_p2.str());

    std::ostringstream intent_p3;
    intent_p3 << header->intent_p3;
    EncapsulateMetaData<std::string>(thisDic, "intent_p3", intent_p3.str());

    std::ostringstream intent_code;
    intent_code << header->intent_code;
    EncapsulateMetaData<std::string>(thisDic, "intent_code", intent_code.str());

    std::ostringstream datatype;
    datatype << header->datatype;
    EncapsulateMetaData<std::string>(thisDic, "datatype", datatype.str());

    std::ostringstream bitpix;
    bitpix << header->bitpix;
    EncapsulateMetaData<std::string>(thisDic, "bitpix", bitpix.str());

    std::ostringstream slice_start;
    slice_start << header->slice_start;
    EncapsulateMetaData<std::string>(thisDic, "slice_start", slice_start.str());

    for (int idx = 0; idx < 8; idx++)
      {
      std::ostringstream pixdim;
      pixdim << header->pixdim[idx];
      std::ostringstream pixdimKey;
      pixdimKey << "pixdim[" << idx << "]";
      EncapsulateMetaData<std::string>(thisDic, pixdimKey.str(), pixdim.str());
      }

    std::ostringstream vox_offset;
    vox_offset << header->vox_offset;
    EncapsulateMetaData<std::string>(thisDic, "vox_offset", vox_offset.str());

    std::ostringstream scl_slope;
    scl_slope << header->scl_slope;
    EncapsulateMetaData<std::string>(thisDic, "scl_slope", scl_slope.str());

    std::ostringstream scl_inter;
    scl_inter << header->scl_inter;
    EncapsulateMetaData<std::string>(thisDic, "scl_inter", scl_inter.str());

    std::ostringstream slice_end;
    slice_end << header->slice_end;
    EncapsulateMetaData<std::string>(thisDic, "slice_end", slice_end.str());

    std::ostringstream slice_code;
    slice_code << header->slice_code;
    EncapsulateMetaData<std::string>(thisDic, "slice_code", slice_code.str());

    std::ostringstream xyzt_units;
    xyzt_units << header->xyzt_units;
    EncapsulateMetaData<std::string>(thisDic, "xyzt_units", xyzt_units.str());

    std::ostringstream cal_max;
    cal_max << header->cal_max;
    EncapsulateMetaData<std::string>(thisDic, "cal_max", cal_max.str());

    std::ostringstream cal_min;
    cal_min << header->cal_min;
    EncapsulateMetaData<std::string>(thisDic, "cal_min", cal_min.str());

    std::ostringstream slice_duration;
    slice_duration << header->slice_duration;
    EncapsulateMetaData<std::string>(thisDic, "slice_duration", slice_duration.str());

    std::ostringstream toffset;
    toffset << header->toffset;
    EncapsulateMetaData<std::string>(thisDic, "toffset", toffset.str());

    std::ostringstream descrip;
    descrip << header->descrip;
    EncapsulateMetaData<std::string>(thisDic, "descrip", descrip.str());

    std::ostringstream aux_file;
    aux_file << header->aux_file;
    EncapsulateMetaData<std::string>(thisDic, "aux_file", aux_file.str());

    std::ostringstream qform_code;
    qform_code << header->qform_code;
    EncapsulateMetaData<std::string>(thisDic, "qform_code", qform_code.str());

    std::ostringstream sform_code;
    sform_code << header->sform_code;
    EncapsulateMetaData<std::string>(thisDic, "sform_code", sform_code.str());

    std::ostringstream quatern_b;
    quatern_b << header->quatern_b;
    EncapsulateMetaData<std::string>(thisDic, "quatern_b", quatern_b.str());

    std::ostringstream quatern_c;
    quatern_c << header->quatern_c;
    EncapsulateMetaData<std::string>(thisDic, "quatern_c", quatern_c.str());

    std::ostringstream quatern_d;
    quatern_d << header->quatern_d;
    EncapsulateMetaData<std::string>(thisDic, "quatern_d", quatern_d.str());

    std::ostringstream qoffset_x;
    qoffset_x << header->qoffset_x;
    EncapsulateMetaData<std::string>(thisDic, "qoffset_x", qoffset_x.str());

    std::ostringstream qoffset_y;
    qoffset_y << header->qoffset_y;
    EncapsulateMetaData<std::string>(thisDic, "qoffset_y", qoffset_y.str());

    std::ostringstream qoffset_z;
    qoffset_z << header->qoffset_z;
    EncapsulateMetaData<std::string>(thisDic, "qoffset_z", qoffset_z.str());

    std::ostringstream srow_x;
    srow_x << header->srow_x[0] << " " << header->srow_x[1] << " " << header->srow_x[2] << " " << header->srow_x[3];
    EncapsulateMetaData<std::string>(thisDic, "srow_x", srow_x.str());

    std::ostringstream srow_y;
    srow_y << header->srow_y[0] << " " << header->srow_y[1] << " " << header->srow_y[2] << " " << header->srow_y[3];
    EncapsulateMetaData<std::string>(thisDic, "srow_y", srow_y.str());

    std::ostringstream srow_z;
    srow_z << header->srow_z[0] << " " << header->srow_z[1] << " " << header->srow_z[2] << " " << header->srow_z[3];
    EncapsulateMetaData<std::string>(thisDic, "srow_z", srow_z.str());

    std::ostringstream intent_name;
    intent_name << header->intent_name;
    EncapsulateMetaData<std::string>(thisDic, "intent_name", intent_name.str());
    free(header);
    }
}

void
NiftiImageIO3201
::ReadImageInformation()
{
  m_NiftiImage=nifti_image_read(this->GetFileName(),false);
  static std::string prev;
  if(prev != this->GetFileName())
    {
#if defined(__USE_VERY_VERBOSE_NIFTI_DEBUGGING__)
    DumpNiftiHeader(this->GetFileName());
#endif
    prev = this->GetFileName();
    }
  if(m_NiftiImage == 0)
    {
    itkExceptionMacro(<< this->GetFileName() << " is not recognized as a NIFTI file");
    }
  //Check the intent code, it is a vector image, or matrix image, then this is not true.
  //
  if(m_NiftiImage->intent_code == NIFTI_INTENT_VECTOR ||
     m_NiftiImage->intent_code == NIFTI_INTENT_SYMMATRIX)
    {
    if(m_NiftiImage->dim[4] > 1)
      {
      this->SetNumberOfDimensions(4);
      }
    else if(m_NiftiImage->dim[3] > 1)
      {
      this->SetNumberOfDimensions(3);
      }
    else if(m_NiftiImage->dim[2] > 1)
      {
      this->SetNumberOfDimensions(2);
      }
    else
      {
      this->SetNumberOfDimensions(1);
      }
    }
  else if(m_NiftiImage->intent_code == NIFTI_INTENT_GENMATRIX)
    { //TODO:  NEED TO DEAL WITH CASE WHERE NIFTI_INTENT_MATRIX
    itkExceptionMacro(<< this->GetFileName() << " has an intent code of NIFTI_INTENT_GENMATRIX which is not yet implemented in ITK");
    }
  else
    { //Simple Scalar Image
    //
    //    this->SetNumberOfDimensions(m_NiftiImage->dim[0]);
    // HACK ALERT KW
    // Apparently some straight-from-the-scanner files report as 4D
    // with T = 1; this causes ImageFileReader to erroneously ignore the reported
    // direction cosines.
    unsigned realdim;
    for(realdim = m_NiftiImage->dim[0]; 
        m_NiftiImage->dim[realdim] == 1 && realdim > 3;
        realdim--)
      {
      }
    this->SetNumberOfDimensions(realdim);
    this->SetNumberOfComponents(1);
    }

  if(m_NiftiImage->intent_code == NIFTI_INTENT_VECTOR ||
     m_NiftiImage->intent_code == NIFTI_INTENT_SYMMATRIX)
    {
    this->SetNumberOfComponents(m_NiftiImage->dim[5]);
    }
  else if(m_NiftiImage->intent_code == NIFTI_INTENT_GENMATRIX)
    { //TODO:  NEED TO DEAL WITH CASE WHERE NIFTI_INTENT_MATRIX
    itkExceptionMacro(<< this->GetFileName() << " has an intent code of NIFTI_INTENT_GENMATRIX which is not yet implemented in ITK");
    }
  //TODO:  Dealing with NIFTI_INTENT_VECTOR or NIFTI_INTENT_GENMATRIX with data type of NIFTI_TYPE_COMPLEX64 NIFTI_TYPE_COMPLEX128 NIFTI_TYPE_RGB24 not supported.

  switch( m_NiftiImage->datatype )
    {
    case NIFTI_TYPE_INT8:
      m_ComponentType = CHAR;
      m_PixelType = SCALAR;
      break;
    case NIFTI_TYPE_UINT8:
      m_ComponentType = UCHAR;
      m_PixelType = SCALAR;
      break;
    case NIFTI_TYPE_INT16:
      m_ComponentType = SHORT;
      m_PixelType = SCALAR;
      break;
    case NIFTI_TYPE_UINT16:
      m_ComponentType = USHORT;
      m_PixelType = SCALAR;
      break;
    case NIFTI_TYPE_INT32:
      m_ComponentType = INT;
      m_PixelType = SCALAR;
      break;
    case NIFTI_TYPE_UINT32:
      m_ComponentType = UINT;
      m_PixelType = SCALAR;
      break;
    case NIFTI_TYPE_FLOAT32:
      m_ComponentType = FLOAT;
      m_PixelType = SCALAR;
      break;
    case NIFTI_TYPE_FLOAT64:
      m_ComponentType = DOUBLE;
      m_PixelType = SCALAR;
      break;
    case NIFTI_TYPE_COMPLEX64:
      m_ComponentType = FLOAT;
      m_PixelType = COMPLEX;
      this->SetNumberOfComponents(2);
      break;
    case NIFTI_TYPE_COMPLEX128:
      m_ComponentType = DOUBLE;
      m_PixelType = COMPLEX;
      this->SetNumberOfComponents(2);
      break;
    case NIFTI_TYPE_RGB24:
      m_ComponentType = UCHAR;
      m_PixelType = RGB;
      this->SetNumberOfComponents(3);
      //TODO:  Need to be able to read/write RGB images into ITK.
      //    case DT_RGB:
      // DEBUG -- Assuming this is a triple, not quad
      //image.setDataType( uiig::DATA_RGBQUAD );
      break;
    case NIFTI_TYPE_RGBA32:
      m_ComponentType = UCHAR;
      m_PixelType = RGBA;
      this->SetNumberOfComponents(4);
      break;
    default:
      break;
    }
  // there are a wide variety of intents we ignore
  // but a few wee need to care about
  switch(m_NiftiImage->intent_code)
    {
    case NIFTI_INTENT_NONE:
      break;
    case NIFTI_INTENT_CORREL:
      break;
    case NIFTI_INTENT_TTEST:
      break;
    case NIFTI_INTENT_FTEST:
      break;
    case NIFTI_INTENT_ZSCORE:
      break;
    case NIFTI_INTENT_CHISQ:
      break;
    case NIFTI_INTENT_BETA:
      break;
    case NIFTI_INTENT_BINOM:
      break;
    case NIFTI_INTENT_GAMMA:
      break;
    case NIFTI_INTENT_POISSON:
      break;
    case NIFTI_INTENT_NORMAL:
      break;
    case NIFTI_INTENT_FTEST_NONC:
      break;
    case NIFTI_INTENT_CHISQ_NONC:
      break;
    case NIFTI_INTENT_LOGISTIC:
      break;
    case NIFTI_INTENT_LAPLACE:
      break;
    case NIFTI_INTENT_UNIFORM:
      break;
    case NIFTI_INTENT_TTEST_NONC:
      break;
    case NIFTI_INTENT_WEIBULL:
      break;
    case NIFTI_INTENT_CHI:
      break;
    case NIFTI_INTENT_INVGAUSS:
      break;
    case NIFTI_INTENT_EXTVAL:
      break;
    case NIFTI_INTENT_PVAL:
      break;
    case NIFTI_INTENT_LOGPVAL:
      break;
    case NIFTI_INTENT_LOG10PVAL:
      break;
    case NIFTI_INTENT_ESTIMATE:
      break;
    case NIFTI_INTENT_LABEL:
      break;
    case NIFTI_INTENT_NEURONAME:
      break;
    case NIFTI_INTENT_GENMATRIX:
      break;
    case NIFTI_INTENT_SYMMATRIX:
      this->SetPixelType(SYMMETRICSECONDRANKTENSOR);
      break;
    case NIFTI_INTENT_DISPVECT:
      break;
    case NIFTI_INTENT_VECTOR:
      this->SetPixelType(VECTOR);
      break;
    case NIFTI_INTENT_POINTSET:
      break;
    case NIFTI_INTENT_TRIANGLE:
      break;
    case NIFTI_INTENT_QUATERNION:
      break;
    case NIFTI_INTENT_DIMLESS:
      break;
    case NIFTI_INTENT_TIME_SERIES:
      break;
    case NIFTI_INTENT_NODE_INDEX:
      break;
    case NIFTI_INTENT_RGB_VECTOR:
      break;
    case NIFTI_INTENT_RGBA_VECTOR:
      break;
    case NIFTI_INTENT_SHAPE:
      break;
    }
  // set slope/intercept
  if(m_NiftiImage->qform_code == 0
     && m_NiftiImage->sform_code == 0)
    {
    m_RescaleSlope = 1;
    m_RescaleIntercept = 0;
    }
  else
    {
    if((m_RescaleSlope = m_NiftiImage->scl_slope) == 0)
      {
      m_RescaleSlope = 1;
      }
    m_RescaleIntercept = m_NiftiImage->scl_inter;
    }

  m_OnDiskComponentType = m_ComponentType;
  //
  // if rescale is necessary, promote type reported
  // to ImageFileReader to float
  if(this->MustRescale())
    {
    if(m_ComponentType == CHAR ||
       m_ComponentType == UCHAR ||
       m_ComponentType == SHORT ||
       m_ComponentType == USHORT ||
       m_ComponentType == INT ||
       m_ComponentType == UINT ||
       m_ComponentType == LONG ||
       m_ComponentType == ULONG)
      {
      m_ComponentType = FLOAT;
      }
    }
  //
  // set up the dimension stuff
  double spacingscale=1.0;//default to mm
  switch(XYZT_TO_SPACE(m_NiftiImage->xyz_units))
    {
    case NIFTI_UNITS_METER:
      spacingscale=1e3;
      break;
    case NIFTI_UNITS_MM:
      spacingscale=1e0;
      break;
    case NIFTI_UNITS_MICRON:
      spacingscale=1e-3;
      break;
    }
  double timingscale=1.0;//Default to seconds
  switch(XYZT_TO_TIME(m_NiftiImage->xyz_units))
    {
    case NIFTI_UNITS_SEC:
      timingscale=1.0;
      break;
    case NIFTI_UNITS_MSEC:
      timingscale=1e-3;
      break;
    case NIFTI_UNITS_USEC:
      timingscale=1e-6;
      break;
    }
  const int dims=this->GetNumberOfDimensions();
  switch(dims)
    {
    case 7:
      this->SetDimensions(6,m_NiftiImage->nw);
      //NOTE: Scaling is not defined in this dimension
      this->SetSpacing(6,m_NiftiImage->dw);
    case 6:
      this->SetDimensions(5,m_NiftiImage->nv);
      //NOTE: Scaling is not defined in this dimension
      this->SetSpacing(5,m_NiftiImage->dv);
    case 5:
      this->SetDimensions(4,m_NiftiImage->nu);
      //NOTE: Scaling is not defined in this dimension
      this->SetSpacing(4,m_NiftiImage->du);
    case 4:
      this->SetDimensions(3,m_NiftiImage->nt);
      this->SetSpacing(3,m_NiftiImage->dt*timingscale);
    case 3:
      this->SetDimensions(2,m_NiftiImage->nz);
      this->SetSpacing(2,m_NiftiImage->dz*spacingscale);
    case 2:
      this->SetDimensions(1,m_NiftiImage->ny);
      this->SetSpacing(1,m_NiftiImage->dy*spacingscale);
    case 1:
      this->SetDimensions(0,m_NiftiImage->nx);
      this->SetSpacing(0,m_NiftiImage->dx*spacingscale);
    }

  this->ComputeStrides();
  //Get Dictionary Information
  //Insert Orientation.
  //Need to encapsulate as much Nifti information as possible here.
  MetaDataDictionary &thisDic=this->GetMetaDataDictionary();
  std::string classname(this->GetNameOfClass());
  EncapsulateMetaData<std::string>(thisDic,ITK_InputFilterName, classname);

  switch(m_ComponentType)
    {
    case CHAR:
      EncapsulateMetaData<std::string>(thisDic,ITK_OnDiskStorageTypeName,
                                       std::string(typeid(char).name()));
      break;
    case UCHAR:
      if(m_PixelType == RGB)
        {
        EncapsulateMetaData<std::string>(thisDic,ITK_OnDiskStorageTypeName,
                                         std::string("RGB"));
        }
      else if(m_PixelType == RGBA)
        {
        EncapsulateMetaData<std::string>(thisDic,ITK_OnDiskStorageTypeName,
                                         std::string("RGBA"));
        }
      else
        {
        EncapsulateMetaData<std::string>(thisDic,ITK_OnDiskStorageTypeName,
                                         std::string(typeid(unsigned char).name()));
        }
      break;
    case SHORT:
      EncapsulateMetaData<std::string>(thisDic,ITK_OnDiskStorageTypeName,
                                       std::string(typeid(short).name()));
      break;
    case USHORT:
      EncapsulateMetaData<std::string>(thisDic,ITK_OnDiskStorageTypeName,
                                       std::string(typeid(unsigned short).name()));
      break;
    case INT:
      EncapsulateMetaData<std::string>(thisDic,ITK_OnDiskStorageTypeName,
                                       std::string(typeid(long).name()));
      break;
    case UINT:
      EncapsulateMetaData<std::string>(thisDic,ITK_OnDiskStorageTypeName,
                                       std::string(typeid(unsigned long).name()));
      break;
    case FLOAT:
      EncapsulateMetaData<std::string>(thisDic,ITK_OnDiskStorageTypeName,
                                       std::string(typeid(float).name()));
      break;
    case DOUBLE:
      EncapsulateMetaData<std::string>(thisDic,ITK_OnDiskStorageTypeName,
                                       std::string(typeid(double).name()));
      break;
      //    case NIFTI_TYPE_RGB24: handled above under UChar
      //    case DT_RGB:
      // DEBUG -- Assuming this is a triple, not quad
      //image.setDataType( uiig::DATA_RGBQUAD );
      //      break;
    default:
      break;
    }
  // set the image orientation
  this->SetImageIOOrientationFromNIfTI(dims);

  // Set the metadata.
  this->SetImageIOMetadataFromNIfTI();

  //Important hist fields
  std::string description(m_NiftiImage->descrip);
  EncapsulateMetaData<std::string>(this->GetMetaDataDictionary(),
                                   ITK_FileNotes,description);

  // We don't need the image anymore
  nifti_image_free(m_NiftiImage);
  m_NiftiImage = 0;
}

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
/**
 * For Nifti this does not write a file, it only fills in the
 * appropriate header information.
 */
void
NiftiImageIO3201
::WriteImageInformation(void)
{
  //  MetaDataDictionary &thisDic=this->GetMetaDataDictionary();
  //
  //
  // First of all we need to not go any further if there's
  // a dimension of the image that won't fit in a 16 bit short.
  for(unsigned int i = 0; i < this->GetNumberOfDimensions(); i++)
    {
    unsigned int curdim(this->GetDimensions(i));
    if(curdim > static_cast<unsigned int>(NumericTraits<short>::max()))
      {
      itkExceptionMacro( << "Dimension(" << i << ") = " << curdim 
                         << " is greater than maximum possible dimension " 
                         << NumericTraits<short>::max() );

      }
    }

  // fill out the image header.
  if(m_NiftiImage == 0)
    {
    m_NiftiImage = nifti_simple_init_nim();
    }
  //
  // set the filename
  std::string FName(this->GetFileName());
  //
  // set the file type
  const char * tempextension=nifti_find_file_extension(FName.c_str());
  if(tempextension == NULL)
    {
    itkExceptionMacro( <<
                       "Bad Nifti file name. No extension found for file: " << FName);
    }
  const std::string ExtensionName( tempextension );
  char *tempbasename=nifti_makebasename(FName.c_str());
  const std::string BaseName(tempbasename);
  free(tempbasename); //Need to clear the extension

  const std::string::size_type ext = ExtensionName.rfind(".gz");
  const bool IsCompressed=(ext == std::string::npos)?false:true;
  if( ( ExtensionName == ".nii" || ExtensionName == ".nii.gz" ) &&
      this->GetUseLegacyModeForTwoFileWriting() == false)
    {
    m_NiftiImage->nifti_type = NIFTI_FTYPE_NIFTI1_1;
    }
  else if ( (ExtensionName == ".nia" ) &&
            this->GetUseLegacyModeForTwoFileWriting() == false)
    {
      m_NiftiImage->nifti_type = NIFTI_FTYPE_ASCII;
    }
  else if(ExtensionName == ".hdr" || ExtensionName == ".img"
       || ExtensionName == ".hdr.gz" || ExtensionName == ".img.gz" )
    { //NOTE: LegacyMode is only valid for header extensions .hdr and .img
    if(this->GetUseLegacyModeForTwoFileWriting() == false)
      {
      // This filter needs to write nifti files in it's default mode
      // , not default to legacy analyze files.
      m_NiftiImage->nifti_type = NIFTI_FTYPE_NIFTI1_2;
      }
    else
      {
      //  If it is desired to write out the nifti variant of
      //  ANALYZE7.5.
      //  NOTE: OREINTATION IS NOT WELL DEFINED IN THIS FORMAT.
      m_NiftiImage->nifti_type = NIFTI_FTYPE_ANALYZE;
      }
    }
  else
    {
    itkExceptionMacro(<< "Bad Nifti file name: " << FName);
    }
    m_NiftiImage->fname = nifti_makehdrname(BaseName.c_str(),m_NiftiImage->nifti_type,false,IsCompressed);
    m_NiftiImage->iname = nifti_makeimgname(BaseName.c_str(),m_NiftiImage->nifti_type,false,IsCompressed);
  //     FIELD         NOTES
  //     -----------------------------------------------------
  //     sizeof_hdr    must be 348
  //     -----------------------------------------------------
  //     dim           dim[0] and dim[1] are always required;
  //                   dim[2] is required for 2-D volumes,
  //                   dim[3] for 3-D volumes, etc.
  m_NiftiImage->nvox = 1;
  //Spacial dims in ITK are given in mm.
  //If 4D assume 4thD is in SECONDS, for all of ITK.
  //NOTE: Due to an ambiguity in the nifti specification, some developers
  // external tools believe that the time units must be set, even if there
  // is only one dataset.  Having the time specified for a purly spatial
  // image has no consequence, so go ahead and set it to seconds.
  m_NiftiImage->xyz_units= static_cast< int >( NIFTI_UNITS_MM | NIFTI_UNITS_SEC );
  m_NiftiImage->dim[7] = m_NiftiImage->nw=1;
  m_NiftiImage->dim[6] = m_NiftiImage->nv=1;
  m_NiftiImage->dim[5] = m_NiftiImage->nu=1;
  m_NiftiImage->dim[4] = m_NiftiImage->nt=1;
  m_NiftiImage->dim[3] = m_NiftiImage->nz=1;
  m_NiftiImage->dim[2] = m_NiftiImage->ny=1;
  m_NiftiImage->dim[1] = m_NiftiImage->nx=1;
  switch(this->GetNumberOfDimensions())
    {
    case 7:
      m_NiftiImage->dim[7] = m_NiftiImage->nw
        = static_cast< int >( this->GetDimensions(6) );
      m_NiftiImage->pixdim[7] = m_NiftiImage->dw =
        static_cast<float>( this->GetSpacing(6) );
      m_NiftiImage->nvox *= m_NiftiImage->dim[7];
    case 6:
      m_NiftiImage->dim[6] = m_NiftiImage->nv
        = this->GetDimensions(5);
      m_NiftiImage->pixdim[6] = m_NiftiImage->dv =
        static_cast<float>( this->GetSpacing(5) );
      m_NiftiImage->nvox *= m_NiftiImage->dim[6];
    case 5:
      m_NiftiImage->dim[5] =
        m_NiftiImage->nu = this->GetDimensions(4);
      m_NiftiImage->pixdim[5] =
        m_NiftiImage->du = static_cast<float>( this->GetSpacing(4) );
      m_NiftiImage->nvox *= m_NiftiImage->dim[5];
    case 4:
      m_NiftiImage->dim[4] = m_NiftiImage->nt
        = this->GetDimensions(3);
      m_NiftiImage->pixdim[4] =
        m_NiftiImage->dt = static_cast<float>( this->GetSpacing(3) );
      m_NiftiImage->nvox *= m_NiftiImage->dim[4];
    case 3:
      m_NiftiImage->dim[3] = m_NiftiImage->nz
        = this->GetDimensions(2);
      m_NiftiImage->pixdim[3] =
        m_NiftiImage->dz = static_cast<float>( this->GetSpacing(2) );
      m_NiftiImage->nvox *= m_NiftiImage->dim[3];
    case 2:
      m_NiftiImage->dim[2] = m_NiftiImage->ny
        = this->GetDimensions(1);
      m_NiftiImage->pixdim[2] =
        m_NiftiImage->dy = static_cast<float>( this->GetSpacing(1) );
      m_NiftiImage->nvox *= m_NiftiImage->dim[2];
    case 1:
      m_NiftiImage->dim[1] = m_NiftiImage->nx
        = this->GetDimensions(0);
      m_NiftiImage->pixdim[1] =
        m_NiftiImage->dx = static_cast<float>( this->GetSpacing(0) );
      m_NiftiImage->nvox *= m_NiftiImage->dim[1];
    }

  const unsigned int numComponents = this->GetNumberOfComponents();

  //TODO:  Also need to check for RGB images where numComponets=3
  if( numComponents > 1
      && !(this->GetPixelType() == COMPLEX
           &&  numComponents == 2)
      && !(this->GetPixelType() == RGB
           && numComponents == 3)
      && !(this->GetPixelType() == RGBA
           && numComponents == 4))
    {
    m_NiftiImage->ndim = 5; //This must be 5 for NIFTI_INTENT_VECTOR images.
    m_NiftiImage->dim[0] = 5; //This must be 5 for NIFTI_INTENT_VECTOR images.
    if(this->GetNumberOfDimensions()> 4)
      {
      itkExceptionMacro(<< "Can not store a vector image of more than 4 dimensions in a Nifti file. Dimension=" << this->GetNumberOfDimensions() );
      }
    //
    // support symmetric matrix type
    if(this->GetPixelType() == ImageIOBase::DIFFUSIONTENSOR3D ||
       this->GetPixelType() == ImageIOBase::SYMMETRICSECONDRANKTENSOR)
      {
      m_NiftiImage->intent_code = NIFTI_INTENT_SYMMATRIX;
      }
    else
      {
      m_NiftiImage->intent_code = NIFTI_INTENT_VECTOR;
      }
    m_NiftiImage->nu =
      m_NiftiImage->dim[5] = this->GetNumberOfComponents();
    if(this->GetNumberOfDimensions() < 4)
      {
      m_NiftiImage->nt =
        m_NiftiImage->dim[4] = 1;
      }
    if(this->GetNumberOfDimensions() < 3)
      {
      m_NiftiImage->nz =
        m_NiftiImage->dim[3] = 1;
      }
    if(this->GetNumberOfDimensions() < 2)
      {
      m_NiftiImage->ny =
        m_NiftiImage->dim[2] = 1;
      }
    if(this->GetNumberOfDimensions() < 1)
      {
      m_NiftiImage->nx =
        m_NiftiImage->dim[1] = 1;
      }
    // Update nvox value because in nifti, vector components are the slowest changing direction, not the fastest.
    m_NiftiImage->nvox *= this->GetNumberOfComponents();
    }
  else
    {
    m_NiftiImage->ndim = this->GetNumberOfDimensions();
    m_NiftiImage->dim[0] = this->GetNumberOfDimensions();
    }

  //     -----------------------------------------------------
  //     datatype      needed to specify type of image data
  //     -----------------------------------------------------
  //     bitpix        should correspond correctly to datatype
  //     -----------------------------------------------------
  switch(this->GetComponentType())
    {
    case UCHAR:
      m_NiftiImage->datatype = NIFTI_TYPE_UINT8;
      m_NiftiImage->nbyper = 1;
      break;
    case CHAR:
      m_NiftiImage->datatype = NIFTI_TYPE_INT8;
      m_NiftiImage->nbyper = 1;
      break;
    case USHORT:
      m_NiftiImage->datatype = NIFTI_TYPE_UINT16;
      m_NiftiImage->nbyper = 2;
      break;
    case SHORT:
      m_NiftiImage->datatype = NIFTI_TYPE_INT16;
      m_NiftiImage->nbyper = 2;
      break;
    case ULONG:
    case UINT:
      m_NiftiImage->datatype = NIFTI_TYPE_UINT32;
      m_NiftiImage->nbyper = 4;
      break;
    case LONG:
    case INT:
      m_NiftiImage->datatype = NIFTI_TYPE_INT32;
      m_NiftiImage->nbyper = 4;
      break;
    case FLOAT:
      m_NiftiImage->datatype = NIFTI_TYPE_FLOAT32;
      m_NiftiImage->nbyper = 4;
      break;
    case DOUBLE:
      m_NiftiImage->datatype = NIFTI_TYPE_FLOAT64;
      m_NiftiImage->nbyper = 8;
      break;
    case UNKNOWNCOMPONENTTYPE:
    default:
      {
      itkExceptionMacro(<<
                        "More than one component per pixel not supported");
      }
    }
  switch(this->GetPixelType())
    {
    case VECTOR: //NOTE: VECTOR is un-rolled by nifti to look like a multi-dimensional scalar image
    case SCALAR:
      break;
    case RGB:
      m_NiftiImage->nbyper *= 3;
      m_NiftiImage->datatype = NIFTI_TYPE_RGB24;
      break;
    case RGBA:
      m_NiftiImage->nbyper *= 4;
      m_NiftiImage->datatype = NIFTI_TYPE_RGBA32;
      break;
    case COMPLEX:
      m_NiftiImage->nbyper *= 2;
      switch(this->GetComponentType())
        {
        case FLOAT:
          m_NiftiImage->datatype = NIFTI_TYPE_COMPLEX64;
          break;
        case DOUBLE:
          m_NiftiImage->datatype = NIFTI_TYPE_COMPLEX128;
          break;
        default:
          {
          itkExceptionMacro(<<
                            "Only float or double precision complex type supported");
          }
        }
      break;
    case SYMMETRICSECONDRANKTENSOR:
    case DIFFUSIONTENSOR3D:
      break;
    case OFFSET:
    case POINT:
    case COVARIANTVECTOR:
    case FIXEDARRAY:
    case MATRIX:
    case UNKNOWNPIXELTYPE:
    default:
          itkExceptionMacro(<<
                            "Can not process this pixel type for writing into nifti");

      break;
    }
  //     -----------------------------------------------------
  //     vox_offset    required for an "n+1" header
  //     -----------------------------------------------------
  //     magic         must be "ni1\0" or "n+1\0"
  //     -----------------------------------------------------
  m_NiftiImage->scl_slope = 1.0f;
  m_NiftiImage->scl_inter = 0.0f;
  this->SetNIfTIOrientationFromImageIO(this->GetNumberOfDimensions(),this->GetNumberOfDimensions()); //TODO: Note both arguments are the same, no need to distinguish between them.
  return;
}


bool  
NiftiImageIO3201::
SupportsDimension(unsigned long dim)
{
  return dim > 1 && dim < 5;
}


void  
NiftiImageIO3201::
SetImageIOOrientationFromNIfTI(unsigned short int dims)
{

  typedef SpatialOrientationAdapter OrientAdapterType;

  //
  // in the case of an Analyze75 file, use old analyze orient method.
  if(m_NiftiImage->qform_code == 0
     && m_NiftiImage->sform_code == 0)
    {
    SpatialOrientationAdapter::DirectionType dir;
    SpatialOrientationAdapter::OrientationType orient;
    switch(m_NiftiImage->analyze75_orient)
      {
      case a75_transverse_unflipped:
        orient = SpatialOrientation::ITK_COORDINATE_ORIENTATION_RPI;
        break;
      case a75_sagittal_unflipped:
        orient = SpatialOrientation::ITK_COORDINATE_ORIENTATION_PIR;
        break;
        // according to analyze documents, you don't see flipped
        // orientation in the wild
      case a75_transverse_flipped:
      case a75_coronal_flipped:
      case a75_sagittal_flipped:
      case a75_orient_unknown:
      case a75_coronal_unflipped:
        orient = SpatialOrientation::ITK_COORDINATE_ORIENTATION_RIP;
        break;
      }
    dir =  OrientAdapterType().ToDirectionCosines(orient);
    m_Origin[0] = m_Origin[1] = 0;
    if(dims > 2)
      {
      m_Origin[2] = 0;
      }
    return;
    }

  // not an Analyze file.
  // scale image data based on slope/intercept
  //
  // qform or sform
  //

  // Use the sform if defined, otherwise use the qform
  mat44 headerOrientationeMatrix;
  if(m_NiftiImage->sform_code > 0)
  {
     headerOrientationeMatrix = m_NiftiImage->sto_xyz;
  }
  else
  {
     headerOrientationeMatrix = m_NiftiImage->qto_xyz;
  }

  // Revert the direction due to the nifti vs ITK different convention
  for (unsigned int i = 0; i < 3; ++i)
  {
     for (unsigned int j = 0; j < 2; ++j)
     {
        headerOrientationeMatrix.m[j][i] *= -1.f;
     }
  }

  // Extract the spacings, set the values and normalise the orientation accordindly
  for (unsigned int i = 0; i < 3; ++i)
  {
     float newSpacing = std::sqrt(
           headerOrientationeMatrix.m[0][i] * headerOrientationeMatrix.m[0][i] +
           headerOrientationeMatrix.m[1][i] * headerOrientationeMatrix.m[1][i] +
           headerOrientationeMatrix.m[2][i] * headerOrientationeMatrix.m[2][i]
           );
     if(newSpacing>0.0)
     {
        headerOrientationeMatrix.m[0][i] /= newSpacing;
        headerOrientationeMatrix.m[1][i] /= newSpacing;
        headerOrientationeMatrix.m[2][i] /= newSpacing;
     }
     if (newSpacing != this->GetSpacing(i))
     {
        this->SetSpacing(i, newSpacing);
     }
  }

  // set origin
  m_Origin[0] = -headerOrientationeMatrix.m[0][3];
  if(dims > 1)
  {
     m_Origin[1] = -headerOrientationeMatrix.m[1][3];
  }
  if(dims > 2)
  {
     m_Origin[2] = headerOrientationeMatrix.m[2][3];
  }

  // Set the direction
  const int max_defined_orientation_dims=(dims > 3)?3:dims;
  for (int i = 0; i < max_defined_orientation_dims; i++){
     std::vector<double> direction(dims,0);
     for (int j = 0; j < max_defined_orientation_dims; j++)
     {
        direction[j] = headerOrientationeMatrix.m[j][i];
     }
     this->SetDirection(i,direction);
  }
}

void 
NiftiImageIO3201::
SetNIfTIOrientationFromImageIO(unsigned short int origdims, unsigned short int dims)
{
    // use NIFTI method 2
    m_NiftiImage->sform_code = NIFTI_XFORM_SCANNER_ANAT;
    m_NiftiImage->qform_code = NIFTI_XFORM_ALIGNED_ANAT;

    // Extract the matrix as stored in the ITK format
    mat44 orientationMatrix;
    orientationMatrix.m[0][0]=orientationMatrix.m[1][1]=1.f;
    orientationMatrix.m[2][2]=orientationMatrix.m[3][3]=1.f;
    orientationMatrix.m[0][1]=orientationMatrix.m[0][2]=orientationMatrix.m[0][3]=0.f;
    orientationMatrix.m[1][0]=orientationMatrix.m[1][2]=orientationMatrix.m[1][3]=0.f;
    orientationMatrix.m[2][0]=orientationMatrix.m[2][1]=orientationMatrix.m[2][3]=0.f;
    orientationMatrix.m[3][0]=orientationMatrix.m[3][1]=orientationMatrix.m[3][2]=0.f;
    const int max_defined_orientation_dims=(dims > 3)?3:dims;

    for (int i = 0; i < max_defined_orientation_dims; i++)
    {
       std::vector<double> direction = this->GetDirection(i);
       for (int j = 0; j < max_defined_orientation_dims; j++)
       {
          orientationMatrix.m[j][i] = direction[j];
       }
    }

    // Revert the direction due to the nifti vs ITK different convention
    for (unsigned int i = 0; i < 3; ++i)
    {
       for (unsigned int j = 0; j < 2; ++j)
       {
          orientationMatrix.m[j][i] *= -1.f;
       }
    }

    // Restore the spacing in the matrix
    for (unsigned int i = 0; i < 3; ++i)
    {
       float spacing = this->GetSpacing(i);
       if(spacing>0.0)
       {
          orientationMatrix.m[0][i] *= spacing;
          orientationMatrix.m[1][i] *= spacing;
          orientationMatrix.m[2][i] *= spacing;
       }
    }

    // set origin
    orientationMatrix.m[0][3] = -m_Origin[0];
    if(dims > 1)
    {
          orientationMatrix.m[1][3] = -m_Origin[1];
    }
    if(dims > 2)
    {
    orientationMatrix.m[2][3] = m_Origin[2];
    }

    // Set the sform matrix and its inverse
    m_NiftiImage->sto_xyz = orientationMatrix;
    m_NiftiImage->sto_ijk = nifti_mat44_inverse(m_NiftiImage->sto_xyz);

    // Compute the closest orthogonal matrix for the qform
    // Note that it might be different from the input nifti image
    // since only one matrix can be stored in the ITK format
    nifti_mat44_to_quatern(orientationMatrix,
                           &(m_NiftiImage->quatern_b),
                           &(m_NiftiImage->quatern_c),
                           &(m_NiftiImage->quatern_d),
                           &(m_NiftiImage->qoffset_x),
                           &(m_NiftiImage->qoffset_y),
                           &(m_NiftiImage->qoffset_z),
                           0,
                           0,
                           0,
                           &(m_NiftiImage->qfac));
    m_NiftiImage->pixdim[0] = m_NiftiImage->qfac;
    m_NiftiImage->qto_xyz = nifti_quatern_to_mat44(m_NiftiImage->quatern_b,
                                                   m_NiftiImage->quatern_c,
                                                   m_NiftiImage->quatern_d,
                                                   m_NiftiImage->qoffset_x,
                                                   m_NiftiImage->qoffset_y,
                                                   m_NiftiImage->qoffset_z,
                                                   this->GetSpacing(0),
                                                   this->GetSpacing(1),
                                                   this->GetSpacing(2),
                                                   m_NiftiImage->qfac);
    m_NiftiImage->qto_ijk = nifti_mat44_inverse(m_NiftiImage->qto_xyz);
}

/**
 * Write the image Information before writing data
 */
void
NiftiImageIO3201
::Write( const void* buffer)
{
  this->WriteImageInformation();
  unsigned int numComponents = this->GetNumberOfComponents();
  if(numComponents == 1 ||
     (numComponents == 2 && this->GetPixelType() == COMPLEX) ||
     (numComponents == 3 && this->GetPixelType() == RGB) ||
     (numComponents == 4 && this->GetPixelType() == RGBA))
    {
    // Need a const cast here so that we don't have to copy the memory
    // for writing.
    m_NiftiImage->data=const_cast<void *>(buffer);
    nifti_image_write(m_NiftiImage);
    m_NiftiImage->data = 0; // if left pointing to data buffer
    // nifti_image_free will try and free this memory
    }
  else  ///Image intent is vector image
    {
    for(unsigned int i = 1; i < 8; i++)
      {
      if(m_NiftiImage->dim[i] == 0)
        {
        m_NiftiImage->dim[i] = 1;
        }
      }
    const unsigned numVoxels =
      m_NiftiImage->dim[1] *
      m_NiftiImage->dim[2] *
      m_NiftiImage->dim[3] *
      m_NiftiImage->dim[4];
    const unsigned buffer_size =
      numVoxels*
      numComponents * //Number of componenets
      m_NiftiImage->nbyper;

    char *nifti_buf = new char[buffer_size];
    const char * const itkbuf = (const char *)buffer;
    // Data must be rearranged to meet nifti organzation.
    // nifti_layout[vec][t][z][y][x] = itk_layout[t][z][y][z][vec]
    const unsigned int rowdist=m_NiftiImage->dim[1];
    const unsigned int slicedist=rowdist*m_NiftiImage->dim[2];
    const unsigned int volumedist=slicedist*m_NiftiImage->dim[3];
    const unsigned int seriesdist=volumedist*m_NiftiImage->dim[4];
    //
    // as per ITK bug 0007485
    // NIfTI is lower triangular, ITK is upper triangular.
    // i.e. if a symmetric matrix is
    // a b c
    // b d e
    // c e f
    // ITK stores it a b c d e f, but NIfTI is a b d c e f
    // so on read, step sequentially through the source vector, but
    // reverse the order of vec[2] and vec[3]
    int *vecOrder;
    if(this->GetPixelType() == ImageIOBase::DIFFUSIONTENSOR3D ||
       this->GetPixelType() == ImageIOBase::SYMMETRICSECONDRANKTENSOR)
      {
        vecOrder = niftk::UpperToLowerOrder(niftk::SymMatDim(numComponents));
      }
    else
      {
      vecOrder = new int[numComponents];
      for(unsigned i = 0; i < numComponents; i++)
        {
        vecOrder[i] = i;
        }
      }
    for(int t = 0; t < m_NiftiImage->dim[4]; t++)
      {
      for(int z = 0; z < m_NiftiImage->dim[3]; z++)
        {
        for(int y = 0; y < m_NiftiImage->dim[2]; y++)
          {
          for(int x = 0; x < m_NiftiImage->dim[1]; x++)
            {
            for(unsigned int c=0;c< numComponents; c++)
              {
              const unsigned int nifti_index=(c*seriesdist+volumedist*t + slicedist*z + rowdist*y + x)*m_NiftiImage->nbyper;
              const unsigned int itk_index=((volumedist*t + slicedist*z + rowdist*y + x)*numComponents + vecOrder[c])*m_NiftiImage->nbyper;
              memcpy(nifti_buf+nifti_index,itkbuf+itk_index,m_NiftiImage->nbyper);
              }
            }
          }
        }
      }
    delete [] vecOrder;
    dumpdata(buffer);
    dumpdata(tobuffer);
    //Need a const cast here so that we don't have to copy the memory for
    //writing.
    m_NiftiImage->data=(void *)nifti_buf;
    nifti_image_write(m_NiftiImage);
    m_NiftiImage->data = 0; // if left pointing to data buffer
    delete [] nifti_buf;
    }
}
} // end namespace itk
