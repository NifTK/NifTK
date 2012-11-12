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

"  <category>Segmentation</category>\n"
"  <title>niftkBreastMaskSegmentationFromMRI</title>\n"
"  <description>Executable to segment left and right breasts from a 3D MR volume.</description>\n"
"  <version>0.1</version>\n"
"  <documentation-url>http://cmic.cs.ucl.ac.uk/home/software/</documentation-url>\n"
"  <license>BSD</license>\n"
"  <contributor>John H. Hipwell, Matt Clarkson and Sebastien Ourselin (UCL)</contributor>\n"

  // The mandatory input parameters
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"  <parameters advanced=\"false\">\n"

"    <label>Mandatory Input Parameters</label>\n"
"    <description>These parameters are the minimum necessary for successful execution of the module.</description>\n"

  // Filename of the input image

"    <image fileExtensions=\"*.nii,*.nii.gz\">\n"
"      <name>fileInputStructural</name>\n"
"      <index>0</index>\n"
"      <description>Filename of the input structural MR image to be segmented.</description>\n"
"      <label>Input structural image</label>\n"
"      <default>required</default>\n"
"      <channel>input</channel>\n"
"    </image>\n"

  // Filename of the output segmented image

"    <image fileExtensions=\"*.nii,*.nii.gz\">\n"
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

"    <image fileExtensions=\"*.nii,*.nii.gz\">\n"
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


  // Smooth the input images

"    <boolean>\n"
"      <name>flgSmooth</name>\n"
"      <longflag>smooth</longflag>\n"
"      <description>Smooth the input images [default: yes].</description>\n"
"      <label>Smooth input images</label>\n"
"      <default>true</default>\n"
"    </boolean>\n"

  // The value at which to threshold the bgnd

"    <float>\n"
"      <name>bgndThresholdProb</name>\n"
"      <longflag>tbg</longflag>\n"
"      <description>The value at which to threshold the bgnd (between 0 and 1) [default: 0.6].</description>\n"
"      <label>Background threshold</label>\n"
"      <default>0.6</default>\n"
"      <constraints>\n"
"        <minimum>0</minimum>\n"
"        <maximum>1</maximum>\n"
"        <step>0.01</step>\n"
"      </constraints>\n"
"    </float>\n"


  // The value at which to threshold the final segmentation

"    <float>\n"
"      <name>finalSegmThreshold</name>\n"
"      <longflag>tsg</longflag>\n"
"      <description>The value at which to threshold the final segmentation (between 0 and 1) [default: 0.45].</description>\n"
"      <label>Final segmentation threshold</label>\n"
"      <default>0.45</default>\n"
"      <constraints>\n"
"        <minimum>0</minimum>\n"
"        <maximum>1</maximum>\n"
"        <step>0.01</step>\n"
"      </constraints>\n"
"    </float>\n"


"  </parameters>\n"


  // Intermediate output images for debugging purposes
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"  <parameters advanced=\"true\">\n"

"    <label>Intermediate output images</label>\n"
"    <description>Intermediate output images for debugging purposes</description>\n"


  // Filename of the output background mask

"    <image fileExtensions=\"*.nii,*.nii.gz\">\n"
"      <name>fileOutputBackground</name>\n"
"      <longflag>obgnd</longflag>\n"
"      <description>Output the background mask.</description>\n"
"      <label>Output background mask</label>\n"
"      <default>BackgroundMask.nii.gz</default>\n"
"      <channel>output</channel>\n"
"    </image>\n"

  // Filename of the output pectoral mask

"    <image fileExtensions=\"*.nii,*.nii.gz\">\n"
"      <name>fileOutputPectoral</name>\n"
"      <longflag>opec</longflag>\n"
"      <description>Output the pectoral mask.</description>\n"
"      <label>Output pectoral mask</label>\n"
"      <default>PectoralMask.nii.gz</default>\n"
"      <channel>output</channel>\n"
"    </image>\n"

  // Filename of the output vtk surface

"    <image fileExtensions=\"*.vtk\">\n"
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
