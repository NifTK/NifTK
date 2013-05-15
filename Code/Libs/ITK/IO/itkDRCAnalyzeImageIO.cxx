/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef ITKDRCANALYZEIMAGEIO_CXX
#define ITKDRCANALYZEIMAGEIO_CXX

#include "itkDRCAnalyzeImageIO.h"
#include <stdio.h>
#include <stdlib.h>
#include <itksys/SystemTools.hxx>
#include <itkByteSwapper.h>
#include <itkSpatialOrientation.h>
#include <itkSpatialOrientationAdapter.h>

#include <itkUCLMacro.h>

namespace itk {

DRCAnalyzeImageIO::DRCAnalyzeImageIO()
{
  // Default to DRC mode.
  m_DRCMode = true;
}

DRCAnalyzeImageIO::~DRCAnalyzeImageIO()
{
}

void DRCAnalyzeImageIO::PrintSelf(std::ostream& os, Indent indent) const {
  Superclass::PrintSelf(os, indent);
  os << indent << "m_DRCMode:" << m_DRCMode << std::endl;
}

/** This should be declared public in base class, but unfortunately isn't, so it's a cut and paste. */
static std::string GetExtension( const std::string& filename )
{
  std::string fileExt(itksys::SystemTools::GetFilenameLastExtension(filename));
  //If the last extension is .gz, then need to pull off 2 extensions.
  //.gz is the only valid compression extension.
  if(fileExt == std::string(".gz"))
    {
    fileExt=itksys::SystemTools::GetFilenameLastExtension(itksys::SystemTools::GetFilenameWithoutLastExtension(filename) );
    fileExt += ".gz";
    }
  //Check that a valid extension was found.
  if(fileExt != ".img.gz" && fileExt != ".img" && fileExt != ".hdr")
    {
    return( "" );
    }
  return( fileExt );
}

/** This should be declared public in base class, but unfortunately isn't, so it's a cut and paste. */
static std::string GetRootName( const std::string& filename )
{
  const std::string fileExt = GetExtension(filename);
  // Create a base filename
  // i.e Image.hdr --> Image
  if( fileExt.length() > 0  //Ensure that an extension was found
      && filename.length() > fileExt.length() //Ensure that the filename does not contain only the extension
      )
    {
    const std::string::size_type it = filename.find_last_of( fileExt );
    const std::string baseName( filename, 0, it-(fileExt.length()-1) );
    return( baseName );
    }
  //Default to return same as input when the extension is nothing (Analyze)
  return( filename );
}

/** This should be declared public in base class, but unfortunately isn't, so it's a cut and paste. */
static std::string GetHeaderFileName( const std::string & filename )
{
  std::string ImageFileName = GetRootName(filename);
  ImageFileName += ".hdr";
  return( ImageFileName );
}

void DRCAnalyzeImageIO::ReadImageInformation() {

  // First do all base class method
  AnalyzeImageIO::ReadImageInformation();

  // Now fix the orientation/direction stuff according to DRC analyze
  if (m_DRCMode) {

	niftkitkDebugMacro("DRCAnalyzeImageIO::ReadImageInformation(): - Providing DRC specific functionality.");


    // Set m_MachineByteOrder to the ByteOrder of the machine
    // Start out with file byte order == system byte order
    // this will be changed if we're reading a file to whatever
    // the file actually contains.
    ImageIOBase::ByteOrder machineByteOrder;
    ImageIOBase::ByteOrder byteOrder;

    if (ByteSwapper<int>::SystemIsBigEndian())
      {
      machineByteOrder = byteOrder = BigEndian;
      }
    else
      {
      machineByteOrder = byteOrder = LittleEndian;
      }

    // So this stuff is copied out of base class, and then modified.
    // First problem is that member variables in base class like m_Hdr are private, so we read header again.
    struct dsr header;

    const std::string headerFileName = GetHeaderFileName( this->GetFileName() );
    std::ifstream inputStream;
    inputStream.open(headerFileName.c_str(), std::ios::in | std::ios::binary);
    if( inputStream.fail())
      {
      itkExceptionMacro("File " << headerFileName << " cannot be read");
      }
    if( ! this->ReadBufferAsBinary( inputStream, (void *)&(header), sizeof(struct dsr) ) )
      {
      itkExceptionMacro("Unexpected end of file");
      }
    inputStream.close();

    // if the machine and file endianess are different
    // perform the byte swapping on it
    byteOrder = this->CheckAnalyzeEndian(header);
    if( machineByteOrder != byteOrder  )
      {
      this->SwapHeaderBytesIfNecessary( byteOrder, &header );
      }

    unsigned int numberOfDimensions = header.dime.dim[0];

    itk::DRCAnalyzeImageIO::ValidDRCAnalyzeOrientationFlags temporient = static_cast<itk::DRCAnalyzeImageIO::ValidDRCAnalyzeOrientationFlags> (header.hist.orient);
    itk::SpatialOrientation::ValidCoordinateOrientationFlags coord_orient;

    switch (temporient) {
    case itk::DRCAnalyzeImageIO::ITK_DRC_ANALYZE_ORIENTATION_RAI_AXIAL:
      coord_orient = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RAI;
      break;
    case itk::DRCAnalyzeImageIO::ITK_DRC_ANALYZE_ORIENTATION_RSP_CORONAL:
      coord_orient = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RSP;
      break;
    case itk::DRCAnalyzeImageIO::ITK_DRC_ANALYZE_ORIENTATION_ASL_SAGITTAL:
      coord_orient = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_ASL;
      break;
    default:
      coord_orient = itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RAI;
      itkWarningMacro("Unknown orientation in file " << m_FileName);
    }
    typedef SpatialOrientationAdapter OrientAdapterType;
    SpatialOrientationAdapter::DirectionType dir = OrientAdapterType().ToDirectionCosines(coord_orient);
    unsigned dims = this->GetNumberOfDimensions();

    // always have at least 3 dimensions for the purposes of setting directions
#define itkAnalzyeImageIO_MINDIMS_IS_THREE ( (dims < 3) ? 3 : dims)
    std::vector<double> dirx(itkAnalzyeImageIO_MINDIMS_IS_THREE, 0), diry(itkAnalzyeImageIO_MINDIMS_IS_THREE, 0), dirz(itkAnalzyeImageIO_MINDIMS_IS_THREE, 0);
#undef itkAnalzyeImageIO_MINDIMS_IS_THREE

    dirx[0] = dir[0][0];
    dirx[1] = dir[1][0];
    dirx[2] = dir[2][0];
    diry[0] = dir[0][1];
    diry[1] = dir[1][1];
    diry[2] = dir[2][1];
    dirz[0] = dir[0][2];
    dirz[1] = dir[1][2];
    dirz[2] = dir[2][2];

    // Anything above 3, set to zero?
    for (unsigned i = 3; i < dims; i++) {
      dirx[i] = diry[i] = dirz[i] = 0;
    }
    this->SetDirection(0, dirx);
    this->SetDirection(1, diry);
    if (numberOfDimensions > 2) {
      this->SetDirection(2, dirz);
    }
  }
  else
  {
	niftkitkDebugMacro("DRCAnalyzeImageIO::ReadImageInformation(): - Providing base class (ITK) functionality from itkAnalyzeImageIO.");
  }
}

ImageIOBase::ByteOrder
DRCAnalyzeImageIO::CheckAnalyzeEndian(const struct dsr &temphdr)
{
  ImageIOBase::ByteOrder returnvalue;
  // Machine and header endianess is same

  // checking hk.extents only is NOT a good idea. Many programs do not set
  // hk.extents correctly. Doing an additional check on hk.sizeof_hdr
  // increases chance of correct result. --Juerg Tschirrin Univeristy of Iowa
  // All properly constructed analyze images should have the extents feild
  // set.  It is part of the file format standard.  While most headers of
  // analyze images are 348 bytes long, The Analyze file format allows the
  // header to have other lengths.
  // This code will fail in the unlikely event that the extents feild is
  // not set (invalid anlyze file anyway) and the header is not the normal
  // size.  Other peices of code have used a heuristic on the image
  // dimensions.  If the Image dimensions is greater
  // than 16000 then the image is almost certainly byte-swapped-- Hans

  const ImageIOBase::ByteOrder systemOrder =
    (ByteSwapper<int>::SystemIsBigEndian()) ? BigEndian : LittleEndian;

  if((temphdr.hk.extents == 16384) || (temphdr.hk.sizeof_hdr == 348))
    {
    returnvalue = systemOrder;
    }
  else
    {
    // File does not match machine
    returnvalue = (systemOrder == BigEndian ) ? LittleEndian : BigEndian;
    }
  return returnvalue;
}

void
DRCAnalyzeImageIO::SwapHeaderBytesIfNecessary( ImageIOBase::ByteOrder& byteOrder, struct dsr * const imageheader )
{
  if ( byteOrder == LittleEndian )
    {
    // NOTE: If machine order is little endian, and the data needs to be
    // swapped, the SwapFromBigEndianToSystem is equivalent to
    // SwapFromSystemToBigEndian.
    ByteSwapper<int>::SwapFromSystemToLittleEndian(
                                                   &imageheader->hk.sizeof_hdr);
    ByteSwapper<int  >::SwapFromSystemToLittleEndian(
                                                     &imageheader->hk.extents );
    ByteSwapper<short int>::SwapFromSystemToLittleEndian(
                                                         &imageheader->hk.session_error );
    ByteSwapper<short int>::SwapRangeFromSystemToLittleEndian(
                                                              &imageheader->dime.dim[0], 8 );
    ByteSwapper<short int>::SwapFromSystemToLittleEndian(
                                                         &imageheader->dime.unused1 );
    ByteSwapper<short int>::SwapFromSystemToLittleEndian(
                                                         &imageheader->dime.datatype );
    ByteSwapper<short int>::SwapFromSystemToLittleEndian(
                                                         &imageheader->dime.bitpix );
    ByteSwapper<short int>::SwapFromSystemToLittleEndian(
                                                         &imageheader->dime.dim_un0 );

    ByteSwapper<float>::SwapRangeFromSystemToLittleEndian(
                                                          &imageheader->dime.pixdim[0],8 );
    ByteSwapper<float>::SwapFromSystemToLittleEndian(
                                                     &imageheader->dime.vox_offset );
    ByteSwapper<float>::SwapFromSystemToLittleEndian(
                                                     &imageheader->dime.roi_scale );
    ByteSwapper<float>::SwapFromSystemToLittleEndian(
                                                     &imageheader->dime.funused1 );
    ByteSwapper<float>::SwapFromSystemToLittleEndian(
                                                     &imageheader->dime.funused2 );
    ByteSwapper<float>::SwapFromSystemToLittleEndian(
                                                     &imageheader->dime.cal_max );
    ByteSwapper<float>::SwapFromSystemToLittleEndian(
                                                     &imageheader->dime.cal_min );
    ByteSwapper<int>::SwapFromSystemToLittleEndian(
                                                   &imageheader->dime.compressed );
    ByteSwapper<int>::SwapFromSystemToLittleEndian(
                                                   &imageheader->dime.verified );
    ByteSwapper<int>::SwapFromSystemToLittleEndian(
                                                   &imageheader->dime.glmax );
    ByteSwapper<int>::SwapFromSystemToLittleEndian(
                                                   &imageheader->dime.glmin );

    ByteSwapper<int>::SwapFromSystemToLittleEndian(
                                                   &imageheader->hist.views );
    ByteSwapper<int>::SwapFromSystemToLittleEndian(
                                                   &imageheader->hist.vols_added );
    ByteSwapper<int>::SwapFromSystemToLittleEndian(
                                                   &imageheader->hist.start_field );
    ByteSwapper<int>::SwapFromSystemToLittleEndian(
                                                   &imageheader->hist.field_skip );
    ByteSwapper<int>::SwapFromSystemToLittleEndian(
                                                   &imageheader->hist.omax );
    ByteSwapper<int>::SwapFromSystemToLittleEndian(
                                                   &imageheader->hist.omin );
    ByteSwapper<int>::SwapFromSystemToLittleEndian(
                                                   &imageheader->hist.smax );
    ByteSwapper<int>::SwapFromSystemToLittleEndian(
                                                   &imageheader->hist.smin );
    }
  else if ( byteOrder == BigEndian )
    {
    //NOTE: If machine order is little endian, and the data needs to be
    // swapped, the SwapFromBigEndianToSystem is equivalent to
    // SwapFromSystemToLittleEndian.
    ByteSwapper<int  >::SwapFromSystemToBigEndian(
                                                  &imageheader->hk.sizeof_hdr );
    ByteSwapper<int  >::SwapFromSystemToBigEndian(
                                                  &imageheader->hk.extents );
    ByteSwapper<short int>::SwapFromSystemToBigEndian(
                                                      &imageheader->hk.session_error );

    ByteSwapper<short int>::SwapRangeFromSystemToBigEndian(
                                                           &imageheader->dime.dim[0], 8 );
    ByteSwapper<short int>::SwapFromSystemToBigEndian(
                                                      &imageheader->dime.unused1 );
    ByteSwapper<short int>::SwapFromSystemToBigEndian(
                                                      &imageheader->dime.datatype );
    ByteSwapper<short int>::SwapFromSystemToBigEndian(
                                                      &imageheader->dime.bitpix );
    ByteSwapper<short int>::SwapFromSystemToBigEndian(
                                                      &imageheader->dime.dim_un0 );

    ByteSwapper<float>::SwapRangeFromSystemToBigEndian(
                                                       &imageheader->dime.pixdim[0],8 );
    ByteSwapper<float>::SwapFromSystemToBigEndian(
                                                  &imageheader->dime.vox_offset );
    ByteSwapper<float>::SwapFromSystemToBigEndian(
                                                  &imageheader->dime.roi_scale );
    ByteSwapper<float>::SwapFromSystemToBigEndian(
                                                  &imageheader->dime.funused1 );
    ByteSwapper<float>::SwapFromSystemToBigEndian(
                                                  &imageheader->dime.funused2 );
    ByteSwapper<float>::SwapFromSystemToBigEndian(
                                                  &imageheader->dime.cal_max );
    ByteSwapper<float>::SwapFromSystemToBigEndian(
                                                  &imageheader->dime.cal_min );
    ByteSwapper<int>::SwapFromSystemToBigEndian(
                                                &imageheader->dime.compressed );
    ByteSwapper<int>::SwapFromSystemToBigEndian(
                                                &imageheader->dime.verified );
    ByteSwapper<int>::SwapFromSystemToBigEndian(
                                                &imageheader->dime.glmax );
    ByteSwapper<int>::SwapFromSystemToBigEndian(
                                                &imageheader->dime.glmin );

    ByteSwapper<int>::SwapFromSystemToBigEndian(
                                                &imageheader->hist.views );
    ByteSwapper<int>::SwapFromSystemToBigEndian(
                                                &imageheader->hist.vols_added );
    ByteSwapper<int>::SwapFromSystemToBigEndian(
                                                &imageheader->hist.start_field );
    ByteSwapper<int>::SwapFromSystemToBigEndian(
                                                &imageheader->hist.field_skip );
    ByteSwapper<int>::SwapFromSystemToBigEndian(
                                                &imageheader->hist.omax );
    ByteSwapper<int>::SwapFromSystemToBigEndian(
                                                &imageheader->hist.omin );
    ByteSwapper<int>::SwapFromSystemToBigEndian(
                                                &imageheader->hist.smax );
    ByteSwapper<int>::SwapFromSystemToBigEndian(
                                                &imageheader->hist.smin );
    }
  else
    {
    itkExceptionMacro("Machine Endian Type Unknown");
    }
}

} // end namespace

#endif // end ITKDRCANALYZEIMAGEIO_CXX
