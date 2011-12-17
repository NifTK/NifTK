#include <iostream>
#include <glob.h>
#include "vtkDCMTKImageReader.h"
#include <vtkStructuredPointsWriter.h>
#include <vtkImageChangeInformation.h>
#include <vtkImageData.h>
#include <vtkDataSetReader.h>
#include <vtkImplicitModeller.h>
#include <vtkTransformFilter.h>
#include <vtkDataObject.h>
//-START--------------------------
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkVTKImageIO.h"
#include "itkVTKImageToImageFilter.h"
//-END----------------------------  
int main(int argc, char **argv) {
    if (argc!=5) { // Usage is editted, added one parameter, changed the name
        cerr << "Usage: dcm2itk <dcmroot> <modfile> <dcm-ImageFile.*> <mod-ImageFile.*>" << endl;
        cerr << "Example: dcm2itk tfl_3d_fast_015_001 mark0-qmz-annotateBreast-r1-normal.mod test-dcm.gipl test-mod.gipl" << endl;
        cerr << "         Reads  tfl_3d_fast_015_001.*.dcm and saves as (for example) test-dcm.gipl" << endl;
        cerr << "         Reads mark0-qmz-annotateBreast-r1-normal.mod and projects on  tfl_3d_fast_015_001.*.dcm2vtk " << endl;
        cerr << "         and saves to test-mod.gipl.  " << endl;
        return -1;
    }
    
    //-START--------------------------
    const unsigned int Dimension = 3;
    typedef unsigned short VolPixelType;
    typedef float ModPixelType;
     
    typedef itk::Image< VolPixelType,  Dimension >  VolImageType;
    typedef itk::Image< ModPixelType,  Dimension >  ModImageType;

    // writer for the output images
    typedef itk::ImageFileWriter< VolImageType >  VolWriterType;
    typedef itk::ImageFileWriter< ModImageType >  ModWriterType;
    typedef itk::VTKImageToImageFilter< VolImageType > VolFilterType;
    typedef itk::VTKImageToImageFilter< ModImageType > ModFilterType;
    //-END----------------------------   

    glob_t globbuf;
    globbuf.gl_offs = 2;
    char globstr[180];

    /*sprintf(globstr, "%s.vol", argv[1]);
    cout << "Looking for vol file " << globstr;
    glob(globstr, GLOB_DOOFFS, NULL, &globbuf);
    if (globbuf.gl_pathc!=1) {
        cout << "ERROR " << endl;
        return -1;
    }
    cout << " OK" << endl;*/

    sprintf(globstr, argv[1]);

    sprintf(globstr, "%s.*.dcm", argv[1]);
    cout << "Looking for DICOM files with pattern " << globstr;
    glob(globstr, GLOB_DOOFFS, NULL, &globbuf);
    if (globbuf.gl_pathc == 0) {
        cout << "ERROR " << endl;
        return -1;
    }
    int N = globbuf.gl_pathc;
    cout << " OK. Found " << globbuf.gl_pathc << endl;
    
    // Read DICOM files using dedicated vtkDCMTKImageReader
    // It returns a VTK Image Data object and a vtkTransform object that correlates the real wordl  coordinates xyz
    // and the image coordinates ijk
    vtkDCMTKImageReader *dcmreader = vtkDCMTKImageReader::New();
    dcmreader->DebugOff();
    dcmreader->SetFilePrefix(argv[1]);
    dcmreader->SetFilePattern("%s.%03d.dcm");
    dcmreader->SetDataExtent(0,-1,0,-1,0,N-1);//(0,-1,0,-1,1,N);//
    dcmreader->UpdateWholeExtent();
    vtkTransform *ijk2xyz = dcmreader->GetTransform();
    
    // Change output origin and spacing of dcmreader output to save VTK files with proper voxel distance
    vtkImageChangeInformation *ici = vtkImageChangeInformation::New();
    ici->SetInput((vtkDataObject *)dcmreader->GetOutput());
    ici->SetOutputOrigin(ijk2xyz->GetPosition());
    ici->SetOutputSpacing(ijk2xyz->GetScale());
    
    //-START--------------------------
    // Write dicom image
    VolFilterType::Pointer volFilter = VolFilterType::New();  

    VolWriterType::Pointer volWriter = VolWriterType::New();

    volFilter->SetInput((vtkImageData *)ici->GetOutput());

    volWriter->SetFileName( argv[3] );

    volFilter->Update();

    // write the output
    volWriter->SetInput( volFilter->GetOutput() );

    try 
    { 
      std::cout << "Writing output vol image... " << std::endl;
      volWriter->Update();
    } 
    catch( itk::ExceptionObject & err ) 
    { 
      std::cerr << "ERROR: ExceptionObject caught !" << std::endl; 
      std::cerr << err << std::endl; 
    }
    //-END----------------------------  

    // Read mod file with 3D model in XYZ coordinates
    vtkDataSetReader *modReader = vtkDataSetReader::New();
    modReader->SetFileName(argv[2]);
    modReader->Update();
    if (!modReader->IsFileUnstructuredGrid()) {
        cout << "File type " << argv[2] << " is not of type vtkUnstructuredGrid" << endl;
        return -1;
    }
    
    // Convert mod xyz to ijk using the current image transform
    // hereby projecting the model on the image
    vtkTransformFilter *modXYZ2IJK = vtkTransformFilter::New();
    modXYZ2IJK->SetInput((vtkDataObject *)modReader->GetOutput());
    modXYZ2IJK->SetTransform(ijk2xyz->GetLinearInverse());
    
    // Transform the ijk model to an image
    vtkImplicitModeller *mod2img = vtkImplicitModeller::New();
    mod2img->SetInput((vtkDataObject *)modXYZ2IJK->GetOutput());
    mod2img->SetCapValue(100);
    mod2img->SetMaximumDistance(0);
    int wext[6],dim[3];
    dcmreader->GetOutput()->GetWholeExtent(wext);
    dcmreader->GetOutput()->GetDimensions(dim);
    mod2img->SetModelBounds((int)wext[0],(int)wext[1],(int)wext[2],(int)wext[3],(int)wext[4],(int)wext[5]);
    mod2img->SetSampleDimensions(dim);

    // Write model image
    ici->SetInput((vtkDataObject *)mod2img->GetOutput());

    //-START--------------------------
    ModWriterType::Pointer modWriter = ModWriterType::New();

    ModFilterType::Pointer modFilter = ModFilterType::New();
  
    ici->Update();
    modFilter->SetInput((vtkImageData *)ici->GetOutput());

    modWriter->SetFileName( argv[4] );

    modFilter->Update();

    // write the output
    modWriter->SetInput( modFilter->GetOutput() );
  
    try 
    { 
      std::cout << "Writing output mod image... " << std::endl;
      modWriter->Update();
    } 
    catch( itk::ExceptionObject & err ) 
    { 
      std::cerr << "ERROR: ExceptionObject caught !" << std::endl; 
      std::cerr << err << std::endl; 
    }
    //-END----------------------------  

    return 0;
}
