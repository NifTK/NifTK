/* $Header: /data/cvsroot/TRUL/vtkTRULLocal/vtkDCMTKImageReader.h,v 1.5 2010/01/05 10:59:37 henkjan Exp $ */
// .NAME vtkDCMTKImageReader - Reads DICOM images using DCMTK library
// Reads multiple DICOM images using a file pattern and stacks them to make one vtkImageData
// Provides GetTag methods to get info on DICOM tags
// Stores all images in internal ImageData scalar array and stores the slice read status
// thus allowing repeated calls with different extents while not having to reread the images.
// 
#ifndef __vtkDCMTKImageReader_h
#define __vtkDCMTKImageReader_h

#include "vtkImageReader2.h"
#include <dcmimage.h>
#include <dcfilefo.h>
#include "vtkTRULLocalConfigure.h" // Include configuration header.
#include "vtkObject.h"
#include "vtkTransform.h"

class VTK_vtkTRULLocal_EXPORT vtkDCMTKImageReader : public vtkImageReader2
{
    public:
    static vtkDCMTKImageReader *New();
    vtkTypeRevisionMacro(vtkDCMTKImageReader,vtkImageReader2);
    
    void PrintSelf(ostream& os, vtkIndent indent);
    
    // Description:
    // Get DICOM Tag item 'pos' as string
    const char *GetTag(const char* tagName, unsigned int pos=0);
    
    // Description:
    // Get DICOM Tag value multiplicity
    // Returns number of positions in tag
    unsigned int GetTagVM(const char* tagName);
    
    // Description:
    // Get DICOM Tag (identified by group (g) and element (e)) as String
    // Content is not interpreted as in GetTag
    // usefull for reading private tags and trying to get content out
    const char *GetTagRawChar(unsigned short g, unsigned short e);
    
    // Description:
    // Get ijk2xyx transform as determined from DICOM file(s).
    // The xyz-pos is obtained from xyz = T->TransformPoint(ijk);
    vtkTransform *GetTransform(void) {return Transform;};
    // Description:
    // Get slice order for multi file DICOM volumes
    // 1  = files are in order to produce positive z scale
    // -1 = files are in reverse order ..
    // otherwise = error due to missing files or perhaps DICOM coding errors
    vtkGetMacro(SliceOrder, double);

    // Description:
    // Set image type
    // Determines how pixel data is interpreted
    // 0 = raw (data as in file)
    // 1 = image (DICOM interpreted using LUTs etc)
    vtkSetMacro(ImageType, int);
    vtkGetMacro(ImageType, int);
    
    
    protected:
    virtual void ExecuteData(vtkDataObject *out);
    virtual int RequestInformation (vtkInformation *, vtkInformationVector** , vtkInformationVector *);

    vtkDCMTKImageReader();
    virtual ~vtkDCMTKImageReader();
    
    private:
    vtkDCMTKImageReader(const vtkDCMTKImageReader&);  // Not implemented.
    void operator=(const vtkDCMTKImageReader&);  // Not implemented.
    // Methods to compute vtkTransform T: xyz = ijk*T
    void ComputeTransformSingleFrame(void);
    void ComputeTransformMultiFrame(void);
    //BTX
    // Description:
    // Read a slice
    void ReadInternalFileName(int slice);
    // Hold DCMTK dataset of each slice
    DcmDataset *dataset;
    // Hold DCMTK fileformat of each slice
    DcmFileFormat *fileformat;
    // Hold DCMTK image of each slice
    DicomImage *image;
    // Hold last requested Tag String
    OFString TagStr;
    int ImageType;
    /// number of rows (in pixel)
    Uint16 Rows;
    /// number of columns (in pixel)
    Uint16 Columns;
    double PixelHeight;
    /// number of bits allocated for each pixel
    Uint16 BitsAllocated;
    /// number of bits stored for each pixel (see 'BitsPerSample')
    Uint16 BitsStored;
    /// position of highest stored bit
    Uint16 HighBit;
    /// number of frames in case of multi-frame images (otherwise '1')
    Uint32 NumberOfFrames;
    /// read status of slice
    bool *sliceRead;
    /// Store for the whole extent of all the slices
    vtkDataArray *slices;
    /// Store for transform
    vtkTransform *Transform;
    double SliceOrder;
    //ETX
};

#endif
