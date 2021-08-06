# Capstone 3 

## Presentation 
https://learnopencv.com/convolutional-neural-network-based-image-colorization-using-opencv/
#### User Uploads an image.
* Read the uploaded image and resize it to what the neural network accepts as its input dimensions.
  * upload = cv.imread("__.png") 
  * W_in, H_in = 255, 2555 
   
##### Convert upload to CIE LAB Color space
 * The input RGB image is scaled so that the values are in the range 0-1, and then it is converted to Lab color space and the lightness channel is extracted out.
  * #Convert the rgb values of the input image to the range of 0 to 1
	* img_rgb = (frame[:,:,[2, 1, 0]] * 1.0 / 255).astype(np.float32)
	* img_lab = cv.cvtColor(img_rgb, cv.COLOR_RGB2Lab)
	* img_l = img_lab[:,:,0] # pull out L channel

The lightness channel in the original image is resized to the network input size which is (224,224) in this case. Usually, the lightness channel ranges from 0 to 100. So we subtract 50 to center it at 0.

  * # resize the lightness channel to network input size
  * img_l_rs = cv.resize(img_l, (W_in, H_in)) # resize image to network input size
  * img_l_rs -= 50 # subtract 50 for mean-centering

#### Read in the Neural Network 
* Read the neural network into memory
  * network = cv.dnn.readNetFromCaffe(protoFile, weightsFile)  

Then we feed the scaled and mean centered lightness channel to the network as its input for the forward pass. The output of the forward pass is the predicted ab channel for the image. It is scaled back to the original image size and then merged with the original sized lightness image(extracted earlier in original resolution) to get the output Lab image. It is then converted to RGB color space to get the final color image. We can then save the output 
image.
  * net.setInput(cv.dnn.blobFromImage(img_l_rs))
	* ab_dec = net.forward()[0,:,:,:].transpose((1,2,0)) # this is our result
	* (H_orig,W_orig) = img_rgb.shape[:2] # original image size
	* ab_dec_us = cv.resize(ab_dec, (W_orig, H_orig))
	* img_lab_out = np.concatenate((img_l[:,:,np.newaxis],ab_dec_us),axis=2) # concatenate with original image L
	* img_bgr_out = np.clip(cv.cvtColor(img_lab_out, cv.COLOR_Lab2BGR), 0, 1)
	* cv.imwrite('dog_colorized.png', cv.resize(img_bgr_out*255, imshowSize))
