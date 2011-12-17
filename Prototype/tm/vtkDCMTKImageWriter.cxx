// $Header: /data/cvsroot/TRUL/vtkTRULLocal/vtkDCMTKImageWriter.cxx,v 1.2 2010/01/05 10:59:37 henkjan Exp $
// C++ Implementation: vtkImageDCMTKImageWriter
#include "vtkDCMTKImageWriter.h"
#include "vtkCommand.h"
#include "vtkErrorCode.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkObjectFactory.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkImageData.h"
// DCMTK stuff
#include <dctk.h>
#include <dcdebug.h>
#include <dcmimage.h>
#include <ofcmdln.h>
#include <dcdatset.h>

vtkCxxRevisionMacro(vtkDCMTKImageWriter, "$Revision: 8041 $");
vtkStandardNewMacro(vtkDCMTKImageWriter);

vtkDCMTKImageWriter::vtkDCMTKImageWriter () {
    if (!dcmDataDict.isDictionaryLoaded()) {
        vtkErrorMacro(<< "Warning: no data dictionary loaded, "
            << "check environment variable: "
            << DCM_DICT_ENVIRONMENT_VARIABLE);
        return;
    }
    this->fileformat = new DcmFileFormat;
    this->dataset = this->fileformat->getDataset();
    this->SOPClassUID=0;
    this->Transform = NULL;
}

vtkDCMTKImageWriter::~vtkDCMTKImageWriter () {
    delete this->fileformat;
}

void vtkDCMTKImageWriter::SetTag(const char* tagName, const char *tagStr) {
    SetTag(this->dataset,tagName,tagStr);
}

void vtkDCMTKImageWriter::SetTag(DcmItem *di, const char* tagName, const char *tagStr) {
    const DcmDataDictionary& globalDataDict = dcmDataDict.rdlock();
    const DcmDictEntry *dicent = globalDataDict.findEntry(tagName);
    OFCondition status = EC_Normal;
    if( dicent == NULL ) {
        vtkErrorMacro(<< "GetTag - Unrecognised tag name: '" << tagName << "' in " << this->InternalFileName);
        dcmDataDict.unlock();
        return;
    }
    dcmDataDict.unlock();
    DcmTagKey tagKey(dicent->getKey());
    status = di->putAndInsertString(tagKey, tagStr);
    if (status.bad()) {
        vtkErrorMacro(<< "Cannot set tagKey " << tagKey << " in " << this->InternalFileName << " because of " << status.text());
        return;
    }
    return;
}
 
int vtkDCMTKImageWriter::RequestData (vtkInformation *request, vtkInformationVector **inputVector, vtkInformationVector *outputVector) {
    this->SetErrorCode(vtkErrorCode::NoError);
    
    vtkInformation *inInfo = inputVector[0]->GetInformationObject(0);
    vtkImageData *input = vtkImageData::SafeDownCast(inInfo->Get(vtkDataObject::DATA_OBJECT()));
  
    // Error checking
    if (input == NULL ) {
        vtkErrorMacro(<<"Write:Please specify an input!");
        return 0;
    }
    if ( !this->FileName && !this->FilePattern) {
        vtkErrorMacro(<<"Write:Please specify either a FileName or a file prefix and pattern");
        this->SetErrorCode(vtkErrorCode::NoFileNameError);
        return 0;
    }
    // Make sure the file name is allocated
    this->InternalFileName = 
        new char[(this->FileName ? strlen(this->FileName) : 1) +
                (this->FilePrefix ? strlen(this->FilePrefix) : 1) +
                (this->FilePattern ? strlen(this->FilePattern) : 1) + 10];
  
    // Fill in image information.
    int *wExt = inInfo->Get(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT());
    this->MinimumFileNumber = this->MaximumFileNumber = this->FileNumber;
    this->FilesDeleted = 0;

    delete this->fileformat;
    this->fileformat = new DcmFileFormat;
    this->dataset = this->fileformat->getDataset();
    // Write
    this->InvokeEvent(vtkCommand::StartEvent);
    this->UpdateProgress(0.0);
    // First fill the tags present in all files
    char uid[100];
    char str[180];
    switch(this->SOPClassUID) {
    case 0:
        this->dataset->putAndInsertString(DCM_SOPClassUID, UID_SecondaryCaptureImageStorage);
        cout << "The DICOM standard requires SecondaryCaptureImageStorage to contain multiframe images. Your doing single frame now." << endl;
        break;
/*    case 1:
        this->dataset->putAndInsertString(DCM_SOPClassUID, UID_MRImageStorage);
        break;*/
    default:
        vtkErrorMacro ( << "Unimplemented or unknown SOPClassUID number" << this->SOPClassUID);
        return 0;
    }
    this->SetTag(this->dataset, "SOPInstanceUID", dcmGenerateUniqueIdentifier(uid, SITE_INSTANCE_UID_ROOT));
    // Error checking
    if (this->Transform == NULL ) {
        vtkErrorMacro(<<"Write:Please specify a transform!");
        return 0;
    }
    int dim[3];
    input->GetDimensions(dim);
    sprintf(str,"%d",dim[0]);
    this->SetTag(this->dataset,"Rows",str);
    sprintf(str,"%d",dim[1]);
    this->SetTag(this->dataset,"Columns",str);
    InsertImagePixelMacro(this->dataset);
    // Now start writing file specific tags and image data
    if (this->FileName) {
        sprintf(this->InternalFileName,"%s",this->FileName);
        this->FileNumber=0;
        this->WriteMultiFrame(input, wExt);
    } else {
        if (this->FilePrefix) {
            for (this->FileNumber = wExt[4]; this->FileNumber <= wExt[5]; this->FileNumber++) {
                sprintf(this->InternalFileName, this->FilePattern, this->FilePrefix, this->FileNumber);
                this->WriteSingleFrame(input, wExt);
            }
        }
    }
    if (this->ErrorCode == vtkErrorCode::OutOfDiskSpaceError) {
        this->DeleteFiles();
    }

    this->UpdateProgress(1.0);
    this->InvokeEvent(vtkCommand::EndEvent);

    delete [] this->InternalFileName;
    this->InternalFileName = NULL;

    return 1;
}

void vtkDCMTKImageWriter::WriteSingleFrame(vtkImageData *input, int *wExt) {
    OFCondition status;
    char str[180];
    sprintf(str,"%d",this->FileNumber);
    this->SetTag(this->dataset,"InstanceNumber",str);
    InsertPixelMeasures(this->dataset);
    InsertImageOrientationPatient(this->dataset);
    InsertImagePositionPatient(this->dataset);
    Uint16 *pixelData = (Uint16 *)input->GetScalarPointer(0,0,this->FileNumber);
    int dim[3];
    input->GetDimensions(dim);
    int pixelLength = dim[0]*dim[1];
    if (dataset->putAndInsertUint16Array(DCM_PixelData, pixelData, pixelLength).bad()) {
        vtkErrorMacro( << "Cannot insert PixelData in SingleFrame " << this->FileNumber);
        return;
    }
    if (this->fileformat->saveFile(this->InternalFileName, EXS_LittleEndianExplicit).bad()) {
        vtkErrorMacro ( << "Cannot write DICOM file ( " << status.text());
        return;
    }
}
void vtkDCMTKImageWriter::WriteMultiFrame(vtkImageData *input, int *wExt) {
    OFCondition status;
    int dim[3];
    input->GetDimensions(dim);
    char str[180];
    sprintf(str,"%d",dim[2]);
    this->SetTag(this->dataset,"NumberOfFrames",str);
    DcmItem *SharedFunctionalGroupsSequence = NULL;
    if (dataset->findOrCreateSequenceItem(DCM_SharedFunctionalGroupsSequence, SharedFunctionalGroupsSequence, 0).bad()) {
        vtkErrorMacro( << "Cannot create SharedFunctionalGroupsSequence");
        return;
    }
    DcmItem *PerFrameFunctionalGroupsSequence = NULL;
    for (this->FileNumber = wExt[4]; this->FileNumber <= wExt[5]; this->FileNumber++) {
        if (dataset->findOrCreateSequenceItem(DCM_PerFrameFunctionalGroupsSequence, PerFrameFunctionalGroupsSequence, this->FileNumber).bad()) {
            vtkErrorMacro( << "Cannot create PerFrameFunctionalGroupsSequence");
            return;
        }
        DcmItem *PixelMeasuresSequence = NULL;
        if (PerFrameFunctionalGroupsSequence->findOrCreateSequenceItem(DCM_PixelMeasuresSequence, PixelMeasuresSequence, 0).bad()) {
            vtkErrorMacro( << "Cannot create PixelMeasuresSequence");
            return;
        }
        InsertPixelMeasures(PixelMeasuresSequence);
        DcmItem *PlanePositionSequence = NULL;
        if (PerFrameFunctionalGroupsSequence->findOrCreateSequenceItem(DCM_PlanePositionSequence, PlanePositionSequence, 0).bad()) {
            vtkErrorMacro( << "Cannot create PlanePositionSequence");
            return;
        }
        InsertImagePositionPatient(PlanePositionSequence);
        DcmItem *PlaneOrientationSequence = NULL;
        if (PerFrameFunctionalGroupsSequence->findOrCreateSequenceItem(DCM_PlaneOrientationSequence, PlaneOrientationSequence, 0).bad()) {
            vtkErrorMacro( << "Cannot create PlaneOrientationSequence");
            return;
        }
        InsertImageOrientationPatient(PlaneOrientationSequence);
    }
    int pixelLength = dim[0]*dim[1]*dim[2];
    Uint16 *pixelData = (Uint16 *)input->GetScalarPointer(0,0,0);
    if (dataset->putAndInsertUint16Array(DCM_PixelData, pixelData, pixelLength).bad()) {
        vtkErrorMacro( << "Cannot insert PixelData in MultiFrame ");
        return;
    }
    if (this->fileformat->saveFile(this->InternalFileName, EXS_LittleEndianExplicit).bad()) {
        vtkErrorMacro ( << "Cannot write DICOM file ( " << status.text());
        return;
    }
}

void vtkDCMTKImageWriter::InsertImagePixelMacro(DcmItem *di) {
    vtkImageData *input = this->GetInput();
    // Pixel module
    switch (input->GetScalarType()) {
    case VTK_UNSIGNED_CHAR:
        this->SetTag(di,"BitsAllocated","8");
        this->SetTag(di,"BitsStored","8");
        this->SetTag(di,"HighBit","7");
        break;
    case VTK_UNSIGNED_SHORT:
        this->SetTag(di,"BitsAllocated","16");
        this->SetTag(di,"BitsStored","16");
        this->SetTag(di,"HighBit","15");
        break;
    default:
        vtkErrorMacro ( << "Cannot handle ScalarType " << input->GetScalarTypeAsString());
        return;
    }
    switch (input->GetNumberOfScalarComponents()) {
    case 1:
        this->SetTag(di,"SamplesPerPixel", "1");
        this->SetTag(di,"PhotometricInterpretation", "MONOCHROME2");
        this->SetTag(di,"PixelRepresentation", "0");
        break;
    case 3:
    case 4:
    default: 
        vtkErrorMacro ( << "Cannot handle NumberOfScalarComponents " << input->GetNumberOfScalarComponents());
        return;
    }
}

void vtkDCMTKImageWriter::InsertPixelMeasures(DcmItem *di) {
    double dx[3];
    char str[80];
    this->Transform->GetScale(dx);
    sprintf(str, "%f\\%f", dx[0], dx[1]);
    this->SetTag(di,"PixelSpacing", str);
    sprintf(str, "%f", dx[2]);
    SetTag(di,"SliceThickness", str);
}

void vtkDCMTKImageWriter::InsertImagePositionPatient(DcmItem *di){
    char str[80];
    float pos[3]={0,0,this->FileNumber};
    this->Transform->TransformPoint(pos,pos);
    sprintf(str, "%f\\%f\\%f", pos[0], pos[1], pos[2]);
    this->SetTag(di,"ImagePositionPatient", str);
}

void vtkDCMTKImageWriter::InsertImageOrientationPatient(DcmItem *di){
    char str[80];
    float colnorm[3]={1,0,0};
    this->Transform->TransformNormal(colnorm, colnorm);
    float rownorm[3]={0,1,0};
    this->Transform->TransformNormal(rownorm, rownorm);
    if (fabs(rownorm[0]*colnorm[0]+rownorm[1]*colnorm[1]+rownorm[2]*colnorm[2])>0.01) {
        vtkErrorMacro(<<"Row and column normal are not perpendicular in transform. Transform invalid.");
        return;
    }
    sprintf(str, "%f\\%f\\%f\\%f\\%f\\%f", colnorm[0], colnorm[1], colnorm[2], rownorm[0], rownorm[1], rownorm[2] );
    this->SetTag(di,"ImageOrientationPatient", str);
}

void vtkDCMTKImageWriter::PrintSelf(ostream& os, vtkIndent indent) {
    this->Superclass::PrintSelf(os,indent);
}
