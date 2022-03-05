Image Enhancement

Contrast stretching is an Image Enhancement method which attempts to improve an image by stretching the range of intensity values.
Here, we stretch the minimum and maximum intensity values present to the possible minimum and maximum intensity values.
for example, If the minimum intensity value(P min ) present in the image is 100 then it is stretched to the possible minimum intensity value 0. Likewise, if the maximum intensity value(P max) is less than the possible maximum intensity value 255 then it is stretched out to 255.

Formula for contrast stretching  :

                            Pnew = (P - Pmin)/(Pmax-Pmin)*255

where Pnew = pixel value of new image
      P    = pixel value of current image
      Pmax = Maximum pixel value
      Pmin = Minimum pixel value

[Note : Contrast stretching is only possible if minimum intensity value and maximum intensity value are not equal to the possible minimum and maximum intensity values. Otherwise, the image generated after contrast stretching will be the same as input image.]
