#! /usr/bin/env python

import os, glob, re, subprocess

def getlabels():
  content='<?xml version="1.0"?>\n\
<document>\n\
<label>\n\
<number>0</number>\n\
<tissueclass>0</tissueclass>\n\
<name>Background and skull</name>\n\
</label>\n\
\n\
<label>\n\
<number>1</number>\n\
<tissueclass>1</tissueclass>\n\
<name>Non-ventricular CSF</name>\n\
</label>\n\
\n\
<label>\n\
<number>5</number>\n\
<tissueclass>1</tissueclass>\n\
<name>3rd Ventricle</name>\n\
</label>\n\
\n\
<label>\n\
<number>12</number>\n\
<tissueclass>1</tissueclass>\n\
<name>4th Ventricle</name>\n\
</label>\n\
\n\
<label>\n\
<number>16</number>\n\
<tissueclass>1</tissueclass>\n\
<name>5th Ventricle</name>\n\
</label>\n\
\n\
<label>\n\
<number>24</number>\n\
<tissueclass>4</tissueclass>\n\
<name>Right Accumbens Area</name>\n\
</label>\n\
\n\
<label>\n\
<number>31</number>\n\
<tissueclass>4</tissueclass>\n\
<name>Left Accumbens Area</name>\n\
</label>\n\
\n\
<label>\n\
<number>32</number>\n\
<tissueclass>4</tissueclass>\n\
<name>Right Amygdala</name>\n\
</label>\n\
\n\
<label>\n\
<number>33</number>\n\
<tissueclass>4</tissueclass>\n\
<name>Left Amygdala</name>\n\
</label>\n\
\n\
<label>\n\
<number>36</number>\n\
<tissueclass>5</tissueclass>\n\
<name>Brain Stem</name>\n\
</label>\n\
\n\
<label>\n\
<number>37</number>\n\
<tissueclass>4</tissueclass>\n\
<name>Right Caudate</name>\n\
</label>\n\
\n\
<label>\n\
<number>38</number>\n\
<tissueclass>4</tissueclass>\n\
<name>Left Caudate</name>\n\
</label>\n\
\n\
<label>\n\
<number>39</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Right Cerebellum Exterior</name>\n\
</label>\n\
\n\
<label>\n\
<number>40</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Left Cerebellum Exterior</name>\n\
</label>\n\
\n\
<label>\n\
<number>41</number>\n\
<tissueclass>3</tissueclass>\n\
<name>Right Cerebellum White Matter</name>\n\
</label>\n\
\n\
<label>\n\
<number>42</number>\n\
<tissueclass>3</tissueclass>\n\
<name>Left Cerebellum White Matter</name>\n\
</label>\n\
\n\
<label>\n\
<number>43</number>\n\
<tissueclass>1</tissueclass>\n\
<name>Right Cerebral Exterior</name>\n\
</label>\n\
\n\
<label>\n\
<number>44</number>\n\
<tissueclass>1</tissueclass>\n\
<name>Left Cerebral Exterior</name>\n\
</label>\n\
\n\
<label>\n\
<number>45</number>\n\
<tissueclass>3</tissueclass>\n\
<name>Right Cerebral White Matter</name>\n\
</label>\n\
\n\
<label>\n\
<number>46</number>\n\
<tissueclass>3</tissueclass>\n\
<name>Left Cerebral White Matter</name>\n\
</label>\n\
\n\
<label>\n\
<number>47</number>\n\
<tissueclass>1</tissueclass>\n\
<name>3rd Ventricle (Posterior part)</name>\n\
</label>\n\
\n\
<label>\n\
<number>48</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Right Hippocampus</name>\n\
</label>\n\
\n\
<label>\n\
<number>49</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Left Hippocampus</name>\n\
</label>\n\
\n\
<label>\n\
<number>50</number>\n\
<tissueclass>1</tissueclass>\n\
<name>Right Inf Lat Vent</name>\n\
</label>\n\
\n\
<label>\n\
<number>51</number>\n\
<tissueclass>1</tissueclass>\n\
<name>Left Inf Lat Vent</name>\n\
</label>\n\
\n\
<label>\n\
<number>52</number>\n\
<tissueclass>1</tissueclass>\n\
<name>Right Lateral Ventricle</name>\n\
</label>\n\
\n\
<label>\n\
<number>53</number>\n\
<tissueclass>1</tissueclass>\n\
<name>Left Lateral Ventricle</name>\n\
</label>\n\
\n\
<label>\n\
<number>54</number>\n\
<tissueclass>4</tissueclass>\n\
<name>Right Lesion</name>\n\
</label>\n\
\n\
<label>\n\
<number>55</number>\n\
<tissueclass>4</tissueclass>\n\
<name>Left Lesion</name>\n\
</label>\n\
\n\
<label>\n\
<number>56</number>\n\
<tissueclass>4</tissueclass>\n\
<name>Right Pallidum</name>\n\
</label>\n\
\n\
<label>\n\
<number>57</number>\n\
<tissueclass>4</tissueclass>\n\
<name>Left Pallidum</name>\n\
</label>\n\
\n\
<label>\n\
<number>58</number>\n\
<tissueclass>4</tissueclass>\n\
<name>Right Putamen</name>\n\
</label>\n\
\n\
<label>\n\
<number>59</number>\n\
<tissueclass>4</tissueclass>\n\
<name>Left Putamen</name>\n\
</label>\n\
\n\
<label>\n\
<number>60</number>\n\
<tissueclass>4</tissueclass>\n\
<name>Right Thalamus Proper</name>\n\
</label>\n\
\n\
<label>\n\
<number>61</number>\n\
<tissueclass>4</tissueclass>\n\
<name>Left Thalamus Proper</name>\n\
</label>\n\
\n\
<label>\n\
<number>62</number>\n\
<tissueclass>3</tissueclass>\n\
<name>Right Ventral DC</name>\n\
</label>\n\
\n\
<label>\n\
<number>63</number>\n\
<tissueclass>3</tissueclass>\n\
<name>Left Ventral DC</name>\n\
</label>\n\
\n\
<label>\n\
<number>64</number>\n\
<tissueclass>4</tissueclass>\n\
<name>Right vessel</name>\n\
</label>\n\
\n\
<label>\n\
<number>65</number>\n\
<tissueclass>4</tissueclass>\n\
<name>Left vessel</name>\n\
</label>\n\
\n\
<label>\n\
<number>70</number>\n\
<tissueclass>4</tissueclass>\n\
<name>Optic Chiasm</name>\n\
</label>\n\
\n\
<label>\n\
<number>72</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Cerebellar Vermal Lobules I-V</name>\n\
</label>\n\
\n\
<label>\n\
<number>73</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Cerebellar Vermal Lobules VI-VII</name>\n\
</label>\n\
\n\
<label>\n\
<number>74</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Cerebellar Vermal Lobules VIII-X</name>\n\
</label>\n\
\n\
<label>\n\
<number>76</number>\n\
<tissueclass>4</tissueclass>\n\
<name>Left Basal Forebrain</name>\n\
</label>\n\
\n\
<label>\n\
<number>77</number>\n\
<tissueclass>4</tissueclass>\n\
<name>Right Basal Forebrain</name>\n\
</label>\n\
\n\
<label>\n\
<number>101</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Right ACgG anterior cingulate gyrus</name>\n\
</label>\n\
\n\
<label>\n\
<number>102</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Left ACgG anterior cingulate gyrus</name>\n\
</label>\n\
\n\
<label>\n\
<number>103</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Right AIns anterior insula</name>\n\
</label>\n\
\n\
<label>\n\
<number>104</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Left AIns anterior insula</name>\n\
</label>\n\
\n\
<label>\n\
<number>105</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Right AOrG anterior orbital gyrus</name>\n\
</label>\n\
\n\
<label>\n\
<number>106</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Left AOrG anterior orbital gyrus</name>\n\
</label>\n\
\n\
<label>\n\
<number>107</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Right AnG angular gyrus</name>\n\
</label>\n\
\n\
<label>\n\
<number>108</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Left AnG angular gyrus</name>\n\
</label>\n\
\n\
<label>\n\
<number>109</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Right Calc calcarine cortex</name>\n\
</label>\n\
\n\
<label>\n\
<number>110</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Left Calc calcarine cortex</name>\n\
</label>\n\
\n\
<label>\n\
<number>113</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Right CO central operculum</name>\n\
</label>\n\
\n\
<label>\n\
<number>114</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Left CO central operculum</name>\n\
</label>\n\
\n\
<label>\n\
<number>115</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Right Cun cuneus</name>\n\
</label>\n\
\n\
<label>\n\
<number>116</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Left Cun cuneus</name>\n\
</label>\n\
\n\
<label>\n\
<number>117</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Right Ent entorhinal area</name>\n\
</label>\n\
\n\
<label>\n\
<number>118</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Left Ent entorhinal area</name>\n\
</label>\n\
\n\
<label>\n\
<number>119</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Right FO frontal operculum</name>\n\
</label>\n\
\n\
<label>\n\
<number>120</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Left FO frontal operculum</name>\n\
</label>\n\
\n\
<label>\n\
<number>121</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Right FRP frontal pole</name>\n\
</label>\n\
\n\
<label>\n\
<number>122</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Left FRP frontal pole</name>\n\
</label>\n\
\n\
<label>\n\
<number>123</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Right FuG fusiform gyrus</name>\n\
</label>\n\
\n\
<label>\n\
<number>124</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Left FuG fusiform gyrus</name>\n\
</label>\n\
\n\
<label>\n\
<number>125</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Right GRe gyrus rectus</name>\n\
</label>\n\
\n\
<label>\n\
<number>126</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Left GRe gyrus rectus</name>\n\
</label>\n\
\n\
<label>\n\
<number>129</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Right IOG inferior occipital gyrus</name>\n\
</label>\n\
\n\
<label>\n\
<number>130</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Left IOG inferior occipital gyrus</name>\n\
</label>\n\
\n\
<label>\n\
<number>133</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Right ITG inferior temporal gyrus</name>\n\
</label>\n\
\n\
<label>\n\
<number>134</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Left ITG inferior temporal gyrus</name>\n\
</label>\n\
\n\
<label>\n\
<number>135</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Right LiG lingual gyrus</name>\n\
</label>\n\
\n\
<label>\n\
<number>136</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Left LiG lingual gyrus</name>\n\
</label>\n\
\n\
<label>\n\
<number>137</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Right LOrG lateral orbital gyrus</name>\n\
</label>\n\
\n\
<label>\n\
<number>138</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Left LOrG lateral orbital gyrus</name>\n\
</label>\n\
\n\
<label>\n\
<number>139</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Right MCgG middle cingulate gyrus</name>\n\
</label>\n\
\n\
<label>\n\
<number>140</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Left MCgG middle cingulate gyrus</name>\n\
</label>\n\
\n\
<label>\n\
<number>141</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Right MFC medial frontal cortex</name>\n\
</label>\n\
\n\
<label>\n\
<number>142</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Left MFC medial frontal cortex</name>\n\
</label>\n\
\n\
<label>\n\
<number>143</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Right MFG middle frontal gyrus</name>\n\
</label>\n\
\n\
<label>\n\
<number>144</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Left MFG middle frontal gyrus</name>\n\
</label>\n\
\n\
<label>\n\
<number>145</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Right MOG middle occipital gyrus</name>\n\
</label>\n\
\n\
<label>\n\
<number>146</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Left MOG middle occipital gyrus</name>\n\
</label>\n\
\n\
<label>\n\
<number>147</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Right MOrG medial orbital gyrus</name>\n\
</label>\n\
\n\
<label>\n\
<number>148</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Left MOrG medial orbital gyrus</name>\n\
</label>\n\
\n\
<label>\n\
<number>149</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Right MPoG postcentral gyrus medial segment</name>\n\
</label>\n\
\n\
<label>\n\
<number>150</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Left MPoG postcentral gyrus medial segment</name>\n\
</label>\n\
\n\
<label>\n\
<number>151</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Right MPrG precentral gyrus medial segment</name>\n\
</label>\n\
\n\
<label>\n\
<number>152</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Left MPrG precentral gyrus medial segment</name>\n\
</label>\n\
\n\
<label>\n\
<number>153</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Right MSFG superior frontal gyrus medial segment</name>\n\
</label>\n\
\n\
<label>\n\
<number>154</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Left MSFG superior frontal gyrus medial segment</name>\n\
</label>\n\
\n\
<label>\n\
<number>155</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Right MTG middle temporal gyrus</name>\n\
</label>\n\
\n\
<label>\n\
<number>156</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Left MTG middle temporal gyrus</name>\n\
</label>\n\
\n\
<label>\n\
<number>157</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Right OCP occipital pole</name>\n\
</label>\n\
\n\
<label>\n\
<number>158</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Left OCP occipital pole</name>\n\
</label>\n\
\n\
<label>\n\
<number>161</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Right OFuG occipital fusiform gyrus</name>\n\
</label>\n\
\n\
<label>\n\
<number>162</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Left OFuG occipital fusiform gyrus</name>\n\
</label>\n\
\n\
<label>\n\
<number>163</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Right OpIFG opercular part of the inferior frontal gyrus</name>\n\
</label>\n\
\n\
<label>\n\
<number>164</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Left OpIFG opercular part of the inferior frontal gyrus</name>\n\
</label>\n\
\n\
<label>\n\
<number>165</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Right OrIFG orbital part of the inferior frontal gyrus</name>\n\
</label>\n\
\n\
<label>\n\
<number>166</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Left OrIFG orbital part of the inferior frontal gyrus</name>\n\
</label>\n\
\n\
<label>\n\
<number>167</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Right PCgG posterior cingulate gyrus</name>\n\
</label>\n\
\n\
<label>\n\
<number>168</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Left PCgG posterior cingulate gyrus</name>\n\
</label>\n\
\n\
<label>\n\
<number>169</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Right PCu precuneus</name>\n\
</label>\n\
\n\
<label>\n\
<number>170</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Left PCu precuneus</name>\n\
</label>\n\
\n\
<label>\n\
<number>171</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Right PHG parahippocampal gyrus</name>\n\
</label>\n\
\n\
<label>\n\
<number>172</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Left PHG parahippocampal gyrus</name>\n\
</label>\n\
\n\
<label>\n\
<number>173</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Right PIns posterior insula</name>\n\
</label>\n\
\n\
<label>\n\
<number>174</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Left PIns posterior insula</name>\n\
</label>\n\
\n\
<label>\n\
<number>175</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Right PO parietal operculum</name>\n\
</label>\n\
\n\
<label>\n\
<number>176</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Left PO parietal operculum</name>\n\
</label>\n\
\n\
<label>\n\
<number>177</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Right PoG postcentral gyrus</name>\n\
</label>\n\
\n\
<label>\n\
<number>178</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Left PoG postcentral gyrus</name>\n\
</label>\n\
\n\
<label>\n\
<number>179</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Right POrG posterior orbital gyrus</name>\n\
</label>\n\
\n\
<label>\n\
<number>180</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Left POrG posterior orbital gyrus</name>\n\
</label>\n\
\n\
<label>\n\
<number>181</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Right PP planum polare</name>\n\
</label>\n\
\n\
<label>\n\
<number>182</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Left PP planum polare</name>\n\
</label>\n\
\n\
<label>\n\
<number>183</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Right PrG precentral gyrus</name>\n\
</label>\n\
\n\
<label>\n\
<number>184</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Left PrG precentral gyrus</name>\n\
</label>\n\
\n\
<label>\n\
<number>185</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Right PT planum temporale</name>\n\
</label>\n\
\n\
<label>\n\
<number>186</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Left PT planum temporale</name>\n\
</label>\n\
\n\
<label>\n\
<number>187</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Right SCA subcallosal area</name>\n\
</label>\n\
\n\
<label>\n\
<number>188</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Left SCA subcallosal area</name>\n\
</label>\n\
\n\
<label>\n\
<number>191</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Right SFG superior frontal gyrus</name>\n\
</label>\n\
\n\
<label>\n\
<number>192</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Left SFG superior frontal gyrus</name>\n\
</label>\n\
\n\
<label>\n\
<number>193</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Right SMC supplementary motor cortex</name>\n\
</label>\n\
\n\
<label>\n\
<number>194</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Left SMC supplementary motor cortex</name>\n\
</label>\n\
\n\
<label>\n\
<number>195</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Right SMG supramarginal gyrus</name>\n\
</label>\n\
\n\
<label>\n\
<number>196</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Left SMG supramarginal gyrus</name>\n\
</label>\n\
\n\
<label>\n\
<number>197</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Right SOG superior occipital gyrus</name>\n\
</label>\n\
\n\
<label>\n\
<number>198</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Left SOG superior occipital gyrus</name>\n\
</label>\n\
\n\
<label>\n\
<number>199</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Right SPL superior parietal lobule</name>\n\
</label>\n\
\n\
<label>\n\
<number>200</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Left SPL superior parietal lobule</name>\n\
</label>\n\
\n\
<label>\n\
<number>201</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Right STG superior temporal gyrus</name>\n\
</label>\n\
\n\
<label>\n\
<number>202</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Left STG superior temporal gyrus</name>\n\
</label>\n\
\n\
<label>\n\
<number>203</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Right TMP temporal pole</name>\n\
</label>\n\
\n\
<label>\n\
<number>204</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Left TMP temporal pole</name>\n\
</label>\n\
\n\
<label>\n\
<number>205</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Right TrIFG triangular part of the inferior frontal gyrus</name>\n\
</label>\n\
\n\
<label>\n\
<number>206</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Left TrIFG triangular part of the inferior frontal gyrus</name>\n\
</label>\n\
\n\
<label>\n\
<number>207</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Right TTG transverse temporal gyrus</name>\n\
</label>\n\
\n\
<label>\n\
<number>208</number>\n\
<tissueclass>2</tissueclass>\n\
<name>Left TTG transverse temporal gyrus</name>\n\
</label>\n\
</document>\n'
  return content
