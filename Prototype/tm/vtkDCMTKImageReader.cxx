/* $Header: /data/cvsroot/TRUL/vtkTRULLocal/vtkDCMTKImageReader.cxx,v 1.6 2010/01/06 12:57:22 henkjan Exp $ */

#include "vtkDCMTKImageReader.h"
#include "vtkImageData.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkObjectFactory.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkPointData.h"
#include "vtkDirectory.h"

#include <vtkstd/vector>
#include <vtkstd/string>

// DCMTK stuff
#include <dctk.h>
#include <dcdebug.h>
#include <dcmimage.h>
#include <ofcmdln.h>
#include <dcdatset.h>
#include <dcfilefo.h>
#include <diutils.h>
#include <dcmimage.h>
#include "djdecode.h"
#include "dipijpeg.h"
#include "dcrledrg.h"
#include "dcpxitem.h"    /* for class DcmPixelItem */
#include "dcpixseq.h"    /* for class DcmPixelSequence */
#include "diinpxt.h"
vtkCxxRevisionMacro(vtkDCMTKImageReader, "$Revision: 8041 $");
vtkStandardNewMacro(vtkDCMTKImageReader);
   
vtkDCMTKImageReader::vtkDCMTKImageReader() {
   this->Debug=false;
   OFBool opt_debug = OFFalse;
   if (this->Debug) {
      opt_debug = OFTrue;
   }
   if (!dcmDataDict.isDictionaryLoaded()) {
      vtkErrorMacro(<< "Warning: no data dictionary loaded, "
            << "check environment variable: "
            << DCM_DICT_ENVIRONMENT_VARIABLE);
      return;
   }
   // register RLE decompression codec
   DcmRLEDecoderRegistration::registerCodecs(OFFalse, opt_debug);
   // register global decompression codecs
   DJDecoderRegistration::registerCodecs(EDC_photometricInterpretation,
                        EUC_default, EPC_default, 
                        opt_debug);  
   if (this->Debug) {
      DicomImageClass::setDebugLevel(DicomImageClass::getDebugLevel() | DicomImageClass::DL_DebugMessages);
   }
   // Let dcmtk report errors
   DicomImageClass::setDebugLevel(DicomImageClass::getDebugLevel() | DicomImageClass::DL_Errors);
   this->fileformat = new DcmFileFormat;
   this->dataset = this->fileformat->getDataset();
   this->image = NULL;
   this->sliceRead = NULL;
   this->slices = NULL;
   this->NumberOfFrames = 1;
   this->Transform = vtkTransform::New();
}

vtkDCMTKImageReader::~vtkDCMTKImageReader() {
   delete this->fileformat;
   delete this->image;
}

void vtkDCMTKImageReader::ReadInternalFileName(int slice) {
   ComputeInternalFileName(slice);
   OFCondition status = EC_Normal;
   if (!this->FileName && !this->FilePattern) {
      vtkErrorMacro("Either a valid FileName or FilePattern must be specified.");
      return;
   }
   delete this->fileformat; // clean up from previous call
   this->fileformat = new DcmFileFormat;
   this->dataset = this->fileformat->getDataset();
   
   status = this->fileformat->loadFile(this->InternalFileName);
   if (status.bad()) {
      vtkErrorMacro(<<status.text()
                  << ": reading file: " << this->InternalFileName);
      return;
   }
   this->fileformat->loadAllDataIntoMemory();
   E_TransferSyntax tsyn = this->dataset->getOriginalXfer();
   // Delete previous image
   if (this->image!=NULL) delete this->image;
   this->image = new DicomImage(this->fileformat, tsyn);
   if (!this->image) {
      vtkErrorMacro(<<"memory exhausted");
      return;
   }
   if (this->image->getStatus() != EIS_Normal) {
      vtkErrorMacro(<< this->InternalFileName << endl << DicomImage::getString(image->getStatus()));
      return;
   }
   return;
}
   
const char *vtkDCMTKImageReader::GetTag(const char* tagName, unsigned int pos) {
   const DcmDataDictionary& globalDataDict = dcmDataDict.rdlock();
   const DcmDictEntry *dicent = globalDataDict.findEntry(tagName);
   DcmElement *elem;
   OFCondition status = EC_Normal;
   if( dicent == NULL ) {
      vtkErrorMacro(<< "GetTag - Unrecognised tag name: '" << tagName << "' in " << this->InternalFileName);
      dcmDataDict.unlock();
      return "";
   }
   dcmDataDict.unlock();
   DcmTagKey tagKey(dicent->getKey());
   this->dataset->findAndGetOFString(tagKey,TagStr,pos);
   status = this->dataset->findAndGetElement(tagKey, elem);
   if (status.bad()) {
      vtkErrorMacro(<< "Cannot find tagKey " << tagKey << " in " << this->InternalFileName << " because of " << status.text());
      return "";
   }
   vtkDebugMacro(<<"GetTag " << tagName << " returning: " << TagStr.data()); 
   return TagStr.data();
}

unsigned int vtkDCMTKImageReader::GetTagVM(const char* tagName) {
   const DcmDataDictionary& globalDataDict = dcmDataDict.rdlock();
   const DcmDictEntry *dicent = globalDataDict.findEntry(tagName);
   DcmElement *elem;
   OFCondition status = EC_Normal;
   if( dicent == NULL ) {
      vtkErrorMacro(<< "GetTagVM - Unrecognised tag name: '" << tagName << "' in " << this->InternalFileName);
      dcmDataDict.unlock();
      return 0;
   }
   dcmDataDict.unlock();
   DcmTagKey tagKey(dicent->getKey());
   status = this->dataset->findAndGetElement(tagKey, elem);
   if (status.bad()) {
      return 0;
   }
   return elem->getVM();
}

const char *vtkDCMTKImageReader::GetTagRawChar(Uint16 g, Uint16 e) {
   // Read the DICOM tag
   OFCondition status = EC_Normal;
   DcmElement *elem;
   //status = this->dataset->findAndGetElement(DcmTagKey(0x0018, 0x0024), elem);
   status = this->dataset->findAndGetElement(DcmTagKey(g, e), elem);
   if (status.bad()) {
      vtkErrorMacro(<< "Cannot find TagKey (" << DcmTagKey(g, e) << ") in " << this->InternalFileName << " because of " << status.text());
      return "";
   }
   // Read the data from the tag as Uint to fool dcmtk to read
   // all data without interpreting string things
   Uint8 *data;
   elem->getUint8Array(data);
   // Get rid of non-printing stuff
   for (int i=0;i<(int)elem->getLength();i++) {
      if (isprint((char)data[i])) {
            // Printable stuff leave untouched
      } else if (((char)data[i]=='\n') ||((char)data[i]=='\r')) {
            // Return character detected
            // Fill with return
            data[i] = (Uint8)'\n';
      } else {
            // Non-printing thing
            // Fill with empy space
            data[i] = (Uint8)' ';
      }
   }
   return (const char *)data;
}

void vtkDCMTKImageReader::ComputeTransformMultiFrame() {
   ReadInternalFileName(0);
   Float64 di, dj, dk;
   // The first item in the PerFrameFunctionalGroupsSequence contains the image info for the first slice
   DcmItem *PerFrameFunctionalGroupsSequenceItem = NULL;
   if (dataset->findAndGetSequenceItem(DCM_PerFrameFunctionalGroupsSequence, PerFrameFunctionalGroupsSequenceItem, 0).bad()) {
      vtkErrorMacro( << "Cannot find PerFrameFunctionalGroupsSequenceItem 0 in " << this->InternalFileName);
      return;
   }
   DcmItem *PixelMeasuresSequenceItem = NULL;
   if (PerFrameFunctionalGroupsSequenceItem->findOrCreateSequenceItem(DCM_PixelMeasuresSequence, PixelMeasuresSequenceItem, 0).bad()) {
      vtkErrorMacro( << "Cannot find PixelMeasuresSequenceItem 0 in " << this->InternalFileName);
      return;
   }
   if (PixelMeasuresSequenceItem->findAndGetFloat64(DCM_PixelSpacing, di, 0).bad()) {
      vtkErrorMacro(<<"Cannot find tag PixelSpacing in " << this->InternalFileName);
   }
   if (PixelMeasuresSequenceItem->findAndGetFloat64(DCM_PixelSpacing, dj, 1).bad()) {
      vtkErrorMacro(<<"Cannot find tag PixelSpacing in " << this->InternalFileName);
   }
   if (PixelMeasuresSequenceItem->findAndGetFloat64(DCM_SliceThickness, dk, 0).bad()) {
      vtkErrorMacro(<<"Cannot find tag  in SliceThickness in " << this->InternalFileName);
   }
   Float64 cx, cy, cz, rx, ry ,rz;
   DcmItem *PlaneOrientationSequenceItem = NULL;
   if (PerFrameFunctionalGroupsSequenceItem->findAndGetSequenceItem(DCM_PlaneOrientationSequence, PlaneOrientationSequenceItem, 0).bad()) {
      vtkErrorMacro( << "Cannot find PlaneOrientationSequenceItem in " << this->InternalFileName);
      return;
   }
   // DICOM definition for  ImageOrientationPatient=(rx,ry,rz,cx,cy,cz)
   if (PlaneOrientationSequenceItem->findAndGetFloat64(DCM_ImageOrientationPatient, rx, 0).bad()) {
      vtkErrorMacro(<<"Cannot find tag ImageOrientationPatient in " << this->InternalFileName);
   }
   if (PlaneOrientationSequenceItem->findAndGetFloat64(DCM_ImageOrientationPatient, ry, 1).bad()) {
      vtkErrorMacro(<<"Cannot find tag ImageOrientationPatient in " << this->InternalFileName);
   }
   if (PlaneOrientationSequenceItem->findAndGetFloat64(DCM_ImageOrientationPatient, rz, 2).bad()) {
      vtkErrorMacro(<<"Cannot find tag ImageOrientationPatient in " << this->InternalFileName);
   }
   if (PlaneOrientationSequenceItem->findAndGetFloat64(DCM_ImageOrientationPatient, cx, 3).bad()) {
      vtkErrorMacro(<<"Cannot find tag ImageOrientationPatient in " << this->InternalFileName);
   }
   if (PlaneOrientationSequenceItem->findAndGetFloat64(DCM_ImageOrientationPatient, cy, 4).bad()) {
      vtkErrorMacro(<<"Cannot find tag ImageOrientationPatient in " << this->InternalFileName);
   }
   if (PlaneOrientationSequenceItem->findAndGetFloat64(DCM_ImageOrientationPatient, cz, 5).bad()) {
      vtkErrorMacro(<<"Cannot find tag ImageOrientationPatient in " << this->InternalFileName);
   }
   if (fabs(1 - sqrt(cx*cx + cy*cy + cz*cz))>0.01) {
      vtkErrorMacro(<<"Column normal not unit length in " << this->InternalFileName);
   }
   if (fabs(1 - sqrt(rx*rx + ry*ry + rz*rz))>0.01) {
      vtkErrorMacro(<<"Row normal not unit length in " << this->InternalFileName);
   }
   if (fabs(rx*cx + ry*cy + rz*cz)>0.01) {
      vtkErrorMacro(<<"Row and column normal not perpendicular in " << this->InternalFileName);
   }
   Float64 sx, sy, sz;
   // Find slice normal by cross product
   sx = cz*ry - cy*rz;
   sy = cx*rz - cz*rx;
   sz = cy*rx - cx*ry;
   DcmItem *PlanePositionSequenceItem = NULL;
   if (PerFrameFunctionalGroupsSequenceItem->findAndGetSequenceItem(DCM_PlanePositionSequence, PlanePositionSequenceItem, 0).bad()) {
      vtkErrorMacro( << "Cannot find PlanePositionSequenceItem in " << this->InternalFileName);
      return;
   }
   Float64 x0, y0, z0;
   if (PlanePositionSequenceItem->findAndGetFloat64(DCM_ImagePositionPatient, x0, 0, OFFalse).bad()) {
      vtkErrorMacro(<<"Cannot find tag ImagePositionPatient in " << this->InternalFileName);
   }
   if (PlanePositionSequenceItem->findAndGetFloat64(DCM_ImagePositionPatient, y0, 1).bad()) {
      vtkErrorMacro(<<"Cannot find tag ImagePositionPatient in " << this->InternalFileName);
   }
   if (PlanePositionSequenceItem->findAndGetFloat64(DCM_ImagePositionPatient, z0, 2).bad()) {
      vtkErrorMacro(<<"Cannot find tag ImagePositionPatient in " << this->InternalFileName);
   }
   double mat[16] = {rx, cx, sx, x0, ry, cy, sy, y0, rz, cz, sz, z0, 0, 0, 0, 1};
   this->Transform->SetMatrix(mat);
   this->Transform->Scale(di,dj,dk); // column width, row width, slicedistance
}

void vtkDCMTKImageReader::ComputeTransformSingleFrame() {
   ReadInternalFileName(0);
   Float64 di, dj, dk;
   if (dataset->findAndGetFloat64(DCM_PixelSpacing, di, 0).bad()) {
      vtkErrorMacro(<<"Cannot find tag PixelSpacing in " << this->InternalFileName);
   }
   if (dataset->findAndGetFloat64(DCM_PixelSpacing, dj, 1).bad()) {
      vtkErrorMacro(<<"Cannot find tag PixelSpacing in " << this->InternalFileName);
   }
   Float64 x0, y0, z0;
   if (dataset->findAndGetFloat64(DCM_ImagePositionPatient, x0, 0, OFFalse).bad()) {
      vtkErrorMacro(<<"Cannot find tag ImagePositionPatient in " << this->InternalFileName);
   }
   if (dataset->findAndGetFloat64(DCM_ImagePositionPatient, y0, 1).bad()) {
      vtkErrorMacro(<<"Cannot find tag ImagePositionPatient in " << this->InternalFileName);
   }
   if (dataset->findAndGetFloat64(DCM_ImagePositionPatient, z0, 2).bad()) {
      vtkErrorMacro(<<"Cannot find tag ImagePositionPatient in " << this->InternalFileName);
   }
   Float64 cx, cy, cz, rx, ry ,rz;
   // DICOM definition for  ImageOrientationPatient=(rx,ry,rz,cx,cy,cz)
   if (dataset->findAndGetFloat64(DCM_ImageOrientationPatient, rx, 0).bad()) {
      vtkErrorMacro(<<"Cannot find tag ImageOrientationPatient in " << this->InternalFileName);
   }
   if (dataset->findAndGetFloat64(DCM_ImageOrientationPatient, ry, 1).bad()) {
      vtkErrorMacro(<<"Cannot find tag ImageOrientationPatient in " << this->InternalFileName);
   }
   if (dataset->findAndGetFloat64(DCM_ImageOrientationPatient, rz, 2).bad()) {
      vtkErrorMacro(<<"Cannot find tag ImageOrientationPatient in " << this->InternalFileName);
   }
   if (dataset->findAndGetFloat64(DCM_ImageOrientationPatient, cx, 3).bad()) {
      vtkErrorMacro(<<"Cannot find tag ImageOrientationPatient in " << this->InternalFileName);
   }
   if (dataset->findAndGetFloat64(DCM_ImageOrientationPatient, cy, 4).bad()) {
      vtkErrorMacro(<<"Cannot find tag ImageOrientationPatient in " << this->InternalFileName);
   }
   if (dataset->findAndGetFloat64(DCM_ImageOrientationPatient, cz, 5).bad()) {
      vtkErrorMacro(<<"Cannot find tag ImageOrientationPatient in " << this->InternalFileName);
   }
   if (fabs(1 - sqrt(cx*cx + cy*cy + cz*cz))>0.01) {
      vtkErrorMacro(<<"Column normal not unit length in " << this->InternalFileName);
   }
   if (fabs(1 - sqrt(rx*rx + ry*ry + rz*rz))>0.01) {
      vtkErrorMacro(<<"Row normal not unit length in " << this->InternalFileName);
   }
   if (fabs(rx*cx + ry*cy + rz*cz)>0.01) {
      vtkErrorMacro(<<"Row and column normal not perpendicular in " << this->InternalFileName);
   }
   Float64 sx, sy, sz;
   // Find slice normal by cross product
   sx = cz*ry - cy*rz;
   sy = cx*rz - cz*rx;
   sz = cy*rx - cx*ry;

   // Multiframe DICOM has all info to get transform in its header
   // For a stack of single slice DICOM files we need to get the last DICOM file
   // to determine the slice thickness and slice order
   if (NumberOfFrames>1) {
      vtkErrorMacro("IMPLEMENT MULTIFRAME GetTRansform !!!!!!!!!!!!!!!");
   } else {
      // Read last file to get last x0,y0,z0
      ReadInternalFileName(this->DataExtent[5]);
      Float64 xe, ye, ze;
      if (dataset->findAndGetFloat64(DCM_ImagePositionPatient, xe, 0).bad()) {
            vtkErrorMacro(<<"Cannot find tag ImagePositionPatient in " << this->InternalFileName);
      }
      if (dataset->findAndGetFloat64(DCM_ImagePositionPatient, ye, 1).bad()) {
            vtkErrorMacro(<<"Cannot find tag ImagePositionPatient in " << this->InternalFileName);
      }
      if (dataset->findAndGetFloat64(DCM_ImagePositionPatient, ze, 2).bad()) {
            vtkErrorMacro(<<"Cannot find tag ImagePositionPatient in " << this->InternalFileName);
      }
      dk = sqrt((xe-x0)*(xe-x0)+(ye-y0)*(ye-y0)+(ze-z0)*(ze-z0))/this->DataExtent[5];
      // Determine slice order and number of slices
      // Compute a predicted xe,ye,ze and determine difference without actual numbers
      Float64 pxe, pye, pze;
      // In vector notation px = x0 + (N-1)*slicethickness*sliceorientation 
      pxe = x0 + this->DataExtent[5]*dk*sx;
      pye = y0 + this->DataExtent[5]*dk*sy;
      pze = z0 + this->DataExtent[5]*dk*sz;
      // slice order
      // If pxe = xe : order = 1;
      // if ||pxe-xe|| = (N-1)*slicethickness order = -1
      // if fabs(fabs(order)-1))<0.1
      SliceOrder = sqrt((pxe-xe)*(pxe-xe)+(pye-ye)*(pye-ye)+(pze-ze)*(pze-ze));
      SliceOrder /= dk*this->DataExtent[5];
      SliceOrder = 1 - SliceOrder;
      if (fabs(fabs(SliceOrder)-1)>(0.1/(dk*this->DataExtent[5]))) {
            vtkErrorMacro(<<"Slice endpoint incorrect. Are there any missing files?" << this->InternalFileName);
      }
   }
   double mat[16] = {rx, cx, sx, x0, ry, cy, sy, y0, rz, cz, sz, z0, 0, 0, 0, 1};
   this->Transform->SetMatrix(mat);
   this->Transform->Scale(di,dj,dk); // column width, row width, slicedistance
}

int vtkDCMTKImageReader::RequestInformation (
   vtkInformation       * vtkNotUsed( request ),
   vtkInformationVector** vtkNotUsed( inputVector ),
   vtkInformationVector * outputVector) 
{
   if (!this->FileName && !this->FilePattern) {
      vtkErrorMacro("Either a valid FileName or FilePattern must be specified.");
      return -1;
   }
   ReadInternalFileName(0);
   // Reset transform
   this->Transform->Identity();

   // Get info from DICOM file
   OFString TmpStr;
   if (dataset->findAndGetOFString(DCM_NumberOfFrames, TmpStr, 0).good()) {
      // A multiframe file
      NumberOfFrames = atoi(TmpStr.data());
      this->DataExtent[5] = NumberOfFrames-1;
   }
   if (dataset->findAndGetUint16(DCM_Rows, Rows).bad()) {
      vtkErrorMacro(<<"Cannot find tag Rows in " << this->InternalFileName);
      Rows = 0;
   }
   if (dataset->findAndGetUint16(DCM_Columns, Columns).bad()) {
      vtkErrorMacro(<<"Cannot find tag Columns in " << this->InternalFileName);
      Columns = 0;
   }
   if (dataset->findAndGetUint16(DCM_BitsAllocated, BitsAllocated).bad()) {
      vtkErrorMacro(<<"Cannot find tag BitsAllocated in " << this->InternalFileName);
      // assume 8 bytes and lets see what's in the image
      BitsAllocated = 8;
   }
   if (dataset->findAndGetUint16(DCM_BitsStored, BitsStored).bad()) {
      vtkErrorMacro(<<"Cannot find tag BitsStored in " << this->InternalFileName);
      // Assume something reasonable and go on
      BitsStored = BitsAllocated;
   }
   if (dataset->findAndGetUint16(DCM_HighBit, HighBit).bad()) {
      vtkErrorMacro(<<"Cannot find tag HighBit in " << this->InternalFileName);
      // Assume something reasonable and go on
      HighBit = BitsAllocated-1;
   }
   Uint16 pixrep;
   if (dataset->findAndGetUint16(DCM_PixelRepresentation, pixrep).bad()) {
      vtkErrorMacro(<<"Cannot find tag PixelRepresentation in " << this->InternalFileName);
   }
   Uint16 nsamp;
   if (dataset->findAndGetUint16(DCM_SamplesPerPixel, nsamp).bad()) {
      vtkErrorMacro(<<"Cannot find tag SamplesPerPixel in " << this->InternalFileName);
   }
   // Determine VTK image settings
   this->DataExtent[0]=0;
   this->DataExtent[1]=Columns-1;
   this->DataExtent[2]=0;
   this->DataExtent[3]=Rows-1;
   vtkDebugMacro(<< "DataExtent: " << this->DataExtent[0] << ", "<< this->DataExtent[1] << ", "<< this->DataExtent[2] << ", "<< this->DataExtent[3] << ", "<< this->DataExtent[4] << ", "<< this->DataExtent[5]);
   switch (BitsAllocated) {
   case 8: {
      if (pixrep==0) {
            this->DataScalarType = VTK_UNSIGNED_CHAR;
      } else {
            this->DataScalarType = VTK_CHAR;
      }
      break;
   }
   case 16: {
      if (pixrep==0) {
            this->DataScalarType = VTK_UNSIGNED_SHORT;
      } else {
            this->DataScalarType = VTK_SHORT;
      }
      break;
   }
   default: {
      vtkErrorMacro(<<"Implement handling " << BitsAllocated << " bit DICOM images" );
      vtkErrorMacro(<< ": reading file: " << this->InternalFileName);
      // Let's assume a common format and see what happens
      if (pixrep == 0) this->DataScalarType = VTK_UNSIGNED_SHORT;
      else this->DataScalarType = VTK_SHORT;
   }
   }
   this->NumberOfScalarComponents=nsamp;

   // Create space to store status of slice being read
   this->sliceRead = (bool *)realloc(this->sliceRead, (this->DataExtent[5]+1) * sizeof(bool));
   //this->sliceRead = (bool *)malloc((this->DataExtent[5]+1) * sizeof(bool));
   for (int i=0; i<=DataExtent[5]; i++ ) {
      this->sliceRead[i] = false;
   }

   // Create a data array to store all the slices
   // The output image data will point to this array at the extent that was requested
   vtkIdType dims[3];
   dims[0] = this->DataExtent[1] - this->DataExtent[0] + 1;
   dims[1] = this->DataExtent[3] - this->DataExtent[2] + 1;
   dims[2] = this->DataExtent[5] - this->DataExtent[4] + 1;
   vtkIdType imageSize = dims[0]*dims[1]*dims[2];
   if (this->slices==NULL) {
      this->slices = vtkDataArray::CreateDataArray(this->DataScalarType);
   }
   this->slices->SetNumberOfComponents(this->NumberOfScalarComponents);
   this->slices->SetNumberOfTuples(imageSize);
   if (NumberOfFrames>1) {
/*      if (!this->FileName) {
            vtkErrorMacro("Can only read multiframe DICOM using SetFileName.");
            return -1;
      }*/
      this->ComputeTransformMultiFrame();
   } else {
      this->ComputeTransformSingleFrame();
   }
   vtkInformation* outInfo = outputVector->GetInformationObject(0);
   outInfo->Set(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(),this->DataExtent, 6);
   vtkDataObject::SetPointDataActiveScalarInfo(outInfo, this->DataScalarType,
      this->NumberOfScalarComponents);
   return 1;

}

void vtkDCMTKImageReader::ExecuteData(vtkDataObject *output) {
   void *ptr = NULL;
   int uext[6], wext[6];
   vtkIdType inc[3];
   vtkImageData *data = vtkImageData::SafeDownCast(output);
   data->GetWholeExtent(wext);
   data->GetUpdateExtent(uext);
   data->GetIncrements(inc);
   ptr = this->slices->GetVoidPointer(uext[4]*(wext[3]-wext[2]+1)*(wext[1]-wext[0]+1));
   unsigned long size = (uext[5]-uext[4]+1)*(wext[3]-wext[2]+1)*(wext[1]-wext[0]+1);
   if (data->GetPointData()->GetScalars()==NULL) {
     // First time allocate scalars object
     data->GetPointData()->SetScalars(vtkDataArray::CreateDataArray(this->DataScalarType));
   }
   data->GetPointData()->GetScalars()->SetVoidArray(ptr,size,1);
   
   // Set extent to update extent using only whole slices
   data->SetExtent(wext[0],wext[1],wext[2],wext[3],uext[4],uext[5]);
   data->GetPointData()->GetScalars()->SetName("DCMTKImageReaderFile");
   this->ComputeDataIncrements();
   
   OFCondition status = EC_Normal;
   vtkDebugMacro("Reading extent: " << wext[0] << ", " << wext[1] << ", " 
            << wext[2] << ", " << wext[3] << ", " << uext[4] << ", " << uext[5]);
   for (int i=uext[4]; i<=uext[5]; i++ ) {
      // If the slice is read than continue
      if (this->sliceRead[i]) continue;
      
      ptr = data->GetScalarPointer(wext[0],wext[2],i);
      ReadInternalFileName(i);
      if (this->ImageType==1) {
            // Read in DICOM interpreted image values
            // set specified window (first window width/center sequence stored in image file). 
            this->image->setWindow(0);
            memcpy(ptr,this->image->getOutputData(),this->DataIncrements[2]);
      } else {
            // Read in raw
            DiInputPixel *InputData;
            unsigned long start = 0;
            unsigned long count = Rows*Columns*this->NumberOfFrames;
            DcmElement *elem;
            status = this->dataset->findAndGetElement(DCM_PixelData, elem);
            if (status.bad()) {
               vtkErrorMacro("No pixel data in " << this->InternalFileName << "Status: " << status.text());
               return;
            }
            DcmPixelData *pixel = OFstatic_cast(DcmPixelData *,elem);
            switch (this->DataScalarType) {
            case VTK_UNSIGNED_CHAR: {
               InputData = new DiInputPixelTemplate<Uint8, Uint8>(pixel, BitsAllocated, BitsStored, HighBit, start, count);
               break;
            }
            case VTK_CHAR: {
               InputData = new DiInputPixelTemplate<Uint8, Sint8>(pixel, BitsAllocated, BitsStored, HighBit, start, count);
               break;
            }
            case VTK_UNSIGNED_SHORT: {
               InputData = new DiInputPixelTemplate<Uint16, Uint16>(pixel, BitsAllocated, BitsStored, HighBit, start, count);
               break;
            }
            case VTK_SHORT: {
               InputData = new DiInputPixelTemplate<Uint16, Sint16>(pixel, BitsAllocated, BitsStored, HighBit, start, count);
               break;
            }
            default: {
               vtkErrorMacro("reading type " << this->DataScalarType << "not implemented");
               return;
            }
            }
            if (InputData == NULL) {
               vtkErrorMacro("insufficient memory");
               return;
            }
           if (InputData->getPixelStart() >= InputData->getCount()) {
               vtkErrorMacro(" start offset exceeds number of pixels stored");
               return;
            }
            memcpy(ptr,InputData->getData(),this->DataIncrements[2]*this->NumberOfFrames);
            delete InputData;
      }
      if (this->NumberOfFrames > 1) {
         // In case of multiframe DICOM all data is read at once
         // and thus all sliceRead==TRUE
         for (unsigned int j=0; j<this->NumberOfFrames; j++) this->sliceRead[j] = true;
      } else {
         // Mark slice i as being read
         this->sliceRead[i] = true;
      }
   }
}

void vtkDCMTKImageReader::PrintSelf(ostream& os, vtkIndent indent) {
   this->Superclass::PrintSelf(os,indent);
   this->fileformat->print(os,DCMTypes::PF_shortenLongTagValues);
   os << indent << "NumberOfFrames: " << this->NumberOfFrames  << "\n";
   os << indent << "SliceOrder: " << this->SliceOrder  << "\n";
}
