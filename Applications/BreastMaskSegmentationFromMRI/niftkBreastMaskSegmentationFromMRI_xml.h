/** @file niftkBreastMaskSegmentationFromMRI_xml.h
 * @date 05/11/2012
 * @author John Hipwell
 * @brief Header file that contains the text to be output
 * for the NifTK command line interface (CLI) module niftkBreastMaskSegmentationFromMRI.
 */


std::string xml_BreastMaskSegmentationFromMRI =
"<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"

  // Executable description
  // ~~~~~~~~~~~~~~~~~~~~~~

"<executable>\n"

"  <category>Breast Cancer Imaging Tools . MRI . Individual images</category>\n"
"  <title>Breast Mask Segmentation from MRI</title>\n"
"  <description>Executable to segment left and right breasts from a 3D MR volume.</description>\n"
"  <version>@NIFTK_VERSION_STRING@</version>\n"
"  <documentation-url>http://cmic.cs.ucl.ac.uk/home/software/</documentation-url>\n"
"  <license>BSD</license>\n"
"  <contributor>John H. Hipwell, Bjorn Eiben (UCL)</contributor>\n"

  // The mandatory input parameters
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"  <parameters advanced=\"false\">\n"

"    <label>Mandatory Input Parameters</label>\n"
"    <description>These parameters are the minimum necessary for successful execution of the module.</description>\n"

  // Filename of the input image

"    <image fileExtensions=\".nii,.nii.gz\">\n"
"      <name>fileInputStructural</name>\n"
"      <index>0</index>\n"
"      <description>Filename of the input structural MR image to be segmented.</description>\n"
"      <label>Input structural image</label>\n"
"      <default>required</default>\n"
"      <channel>input</channel>\n"
"    </image>\n"

  // Filename of the output segmented image

"    <image fileExtensions=\".nii,.nii.gz\">\n"
"      <name>fileOutputImage</name>\n"
"      <flag>o</flag>\n"
"      <description>Filename of the output segmented image.</description>\n"
"      <label>Output segmented image</label>\n"
"      <default>SegmentedImage.nii</default>\n"
"      <channel>output</channel>\n"
"    </image>\n"


"  </parameters>\n"


  // General Options
  // ~~~~~~~~~~~~~~~

"  <parameters advanced=\"true\">\n"

"    <label>General Options</label>\n"
"    <description>Optional input parameters</description>\n"

  // Filename of an optional additional Fat-saturated input MR image

"    <image fileExtensions=\".nii,.nii.gz\">\n"
"      <name>fileInputFatSat</name>\n"
"      <longflag>fs</longflag>\n"
"      <description>An additional optional fat-saturated image (must be the same size and resolution as the structural image).</description>\n"
"      <label>Fat-saturated MRI</label>\n"
"      <default></default>\n"
"      <channel>input</channel>\n"
"    </image>\n"

  // Verbose output

"    <boolean>\n"
"      <name>flgVerbose</name>\n"
"      <flag>v</flag>\n"
"      <description>The level of information printed during execution of the module [default: no].</description>\n"
"      <label>Verbose output</label>\n"
"      <default>false</default>\n"
"    </boolean>\n"


  // Clip the segmented region with fitted B-Spline surface

"    <boolean>\n"
"      <name>flgCropFit</name>\n"
"      <longflag>cropfit</longflag>\n"
"      <description>A B-spline fit to the anterior breast surface is created and used to clip the segmentation [default: no].</description>\n"
"      <label>Clip segmentation with fitted surface</label>\n"
"      <default>false</default>\n"
"    </boolean>\n"


"  </parameters>\n"


  // Intermediate output images for debugging purposes
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"  <parameters advanced=\"true\">\n"

"    <label>Intermediate output images</label>\n"
"    <description>Intermediate output images</description>\n"


  // Filename of the output background mask

"    <image fileExtensions=\".nii,.nii.gz\">\n"
"      <name>fileOutputBackground</name>\n"
"      <longflag>obgnd</longflag>\n"
"      <description>Output the background mask.</description>\n"
"      <label>Output background mask</label>\n"
"      <default>BackgroundMask.nii.gz</default>\n"
"      <channel>output</channel>\n"
"    </image>\n"


  // Filename of the output pectoral mask

"    <image fileExtensions=\".nii,.nii.gz\">\n"
"      <name>fileOutputPectoral</name>\n"
"      <longflag>opec</longflag>\n"
"      <description>Output the pectoral mask.</description>\n"
"      <label>Output pectoral mask</label>\n"
"      <default>PectoralMask.nii.gz</default>\n"
"      <channel>output</channel>\n"
"    </image>\n"



  // Filename of the output vtk surface

"    <image fileExtensions=\".vtk\">\n"
"      <name>fileOutputSurface</name>\n"
"      <longflag>ovtk</longflag>\n"
"      <description>Write the breast surface to a VTK polydata file.</description>\n"
"      <label>Output VTK surface</label>\n"
"      <default>BreastSurface.vtk</default>\n"
"      <channel>output</channel>\n"
"    </image>\n"

"  </parameters>\n"


"</executable>\n"
;
