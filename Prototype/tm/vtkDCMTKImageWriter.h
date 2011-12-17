// $Header: /data/cvsroot/TRUL/vtkTRULLocal/vtkDCMTKImageWriter.h,v 1.2 2010/01/05 10:59:37 henkjan Exp $
// A class to estimate window and level in an image
#ifndef __vtkDCMTKImageWriter_h
#define __vtkDCMTKImageWriter_h

#include "vtkTRULLocalConfigure.h" // Include configuration header.
#include "vtkImageWriter.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkTransform.h"
#include <dcmimage.h>
#include <dcfilefo.h>
#include <dcdatset.h>

class VTK_vtkTRULLocal_EXPORT vtkDCMTKImageWriter : public vtkImageWriter 
{
public:
    // Description:
    // Definition for the vtkStandardNewMacro
    static vtkDCMTKImageWriter *New();

    // Description:
    // Defines SafeDownCast and many other methods
    vtkTypeRevisionMacro(vtkDCMTKImageWriter,vtkImageWriter);

    // Description:
    // Print status
    void PrintSelf(ostream& os, vtkIndent indent);

    // Description:
    // Set SOPClassUID
    // e.g. SetSOPClassUID(0) -> SecondaryCaptureImageStorage
    vtkSetMacro(SOPClassUID, int);
    void vtkSetSOPClassUIDToSecondaryCaptureImageStorage(void) {SOPClassUID=0;};
    void vtkSetSOPClassUIDToMRImageStorage(void) {SOPClassUID=1;};

    // Description:
    // Set DICOM Tag item as string
    // For multivalue tags format string as: "val1//val2// etc"
    void SetTag(DcmItem *di, const char* tagName, const char *tagStr);
    // Short hand call for SetTag where di = this->dataset
    // a root level tag
    void SetTag(const char* tagName, const char *tagStr);

    // Description:
    // Set ijk2xyx transform see btkDCMTKImageReader for details
    // The xyz-pos is obtained from xyz = T->TransformPoint(ijk);
    void SetTransform(vtkTransform *t) {Transform = t;};

    // Description:
    // Set slice order for multi file DICOM volumes
    // 1  = files are in order to produce positive z scale
    // -1 = files are in reverse order
    vtkSetMacro(SliceOrder, double);
protected:
    vtkDCMTKImageWriter();
    ~vtkDCMTKImageWriter();
    virtual int RequestData (vtkInformation *request, vtkInformationVector **inputVector, vtkInformationVector *outputVector);
private:
    void WriteSingleFrame(vtkImageData *input, int *wExt);
    void WriteMultiFrame(vtkImageData *input, int *wExt);
    // Write image pixel macro tags into DcmItem
    void InsertImagePixelMacro(DcmItem *di);
    // Write PixelMeasures tags into DcmItem
    void InsertPixelMeasures(DcmItem *di);
    void InsertImagePositionPatient(DcmItem *di);
    void InsertImageOrientationPatient(DcmItem *di);
    // Hold DCMTK dataset of each slice
    DcmDataset *dataset;
    // Hold DCMTK fileformat of each slice
    DcmFileFormat *fileformat;
    // Hold SOPClassUID number
    int SOPClassUID;
    /// Store for transform
    vtkTransform *Transform;
    double SliceOrder;
};

#endif
