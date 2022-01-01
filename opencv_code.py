#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import cv2


# In[23]:


image = cv2.imread('C:/Users/hoyaj/Downloads/Sample_data (1).png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# In[38]:


plt.imshow(image)
plt.show()


# In[36]:


plt.figure(figsize=(20, 20))
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(image)


# Create our shapening kernel, we don't normalize since the 
# the values in the matrix sum to 1
kernel_sharpening = np.array([[-1,-1,-1], 
                              [-1,9,-1], 
                              [-1,-1,-1]])

# applying different kernels to the input image
sharpened = cv2.filter2D(image, -1, kernel_sharpening)


plt.subplot(1, 2, 2)
plt.title("Image Sharpening")
plt.imshow(sharpened)


# In[8]:


print(image.shape)


# In[34]:


y=30; x=0; h=45; w=480        # 좌표 지정
roi = image[y:y+h, x:x+w,]         # roi 지정       
print(roi.shape)                # roi shape (50,50,3)


# In[35]:


plt.imshow(roi)
plt.show()

