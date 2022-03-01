# Captionista (Image-Caption-Generator)

## We speak the language of images 

TABLE OF CONTENTS:

1.0: Introduction

1.1: Goal

1.2: How does it work

1.3: Use Cases

1.4: Problem Statement

1.5: Dataset Description

1.6: Example

2.0: Image Processing

3.0: Caption Processing

3.1: Caption Cleaning

3.2: Train Test Split

3.3: Tokenizer

3.4: Sequence Creation

4.0: Models

4.1: Model Architecture

4.2: Model Evaluation

4.3: Prediction on unseen images

5.0: Conclusion

6.0: References


 
### 1.0 INTRODUCTION 

Image Captioning is the process of generating a textual description for given images. It has been a very important and fundamental task in the Deep Learning domain. Image captioning has a huge amount of application. 

Image captioning can be regarded as an end-to-end Sequence to Sequence problem, as it converts images, which is regarded as a sequence of pixels to a sequence of words. For this purpose, we need to process both the language or statements and the images. For the Language part, we use recurrent Neural Networks and for the Image part, we use Convolutional Neural Networks to obtain the feature vectors respectively. 

### GOAL 

Our goal with this project is to create a system that can generate captions from an image. So, in the same way, when we look at images and can describe what it contains with words, our system would try to do the same thing. 

 

 ### 1.2 HOW DOES IT WORK? 

Let’s have a look at a simple image: 

![image](https://user-images.githubusercontent.com/62516990/156241683-bf27483d-a821-4b3f-8fa5-e1f828d331f1.png)

Figure 1: Albert Einstein 

 

What’s the first thing that comes to our mind? 

Some answers could be: 

- Famous Scientist  

- Albert Einstein  

- A man standing 

- One of the smartest people of all time  

- Radical thinker  

 

While forming the description, we are seeing the image but at the same time, we are looking to create a meaningful sequence of words. The first part is handled by CNNs and the second is handled by RNNs. 

 

If we can obtain a suitable dataset with images and their corresponding human descriptions, we can train networks to automatically caption images. 

 
### 1.3 USE CASES: 

- Can be used to extract insights and create summaries from images the same way it is done with text 

- A huge benefit for blind people and they would be able to have a better understanding of the world around them 

- The captions can be converted to audio, and this can be useful for podcasts, to describe images 

- Can be used in hospitals to aid clinical physicians to infer conclusions for certain medial images 

 

###  1.4 PROBLEM STATEMENT: 

To parse an image and output a few lines of text that describes what the image contains. 

 

###  1.5 DATASET DESCRIPTION: 

A new benchmark collection for sentence-based image description and search, consisting of 8,000 images that are each paired with five different captions which provide clear descriptions of the salient entities and events. The images were chosen from six different Flickr groups and tend not to contain any well-known people or locations but were manually selected to depict a variety of scenes and situations. 

 
 
###  1.6 EXAMPLE: 

This is a simple example of what our data looks like: 

 

The image: 

![image](https://user-images.githubusercontent.com/62516990/156242050-e210781d-7f5b-4a2e-a6c5-f4660ef4353e.png)


Figure 2: Test Dataset 

 

The captions: 

 

- A man lays on a bench while his dog sits by him 

- A man lays on the bench to which a white dog is also tied  

- a man sleeping on a bench outside with a white and black dog sitting next to him 

- A shirtless man lies on a park bench with his dog 

- A man lying on bench holding leash of dog sitting on ground 

 
 
 
 
 

### 2.0 IMAGE PROCESSING 

The dataset is obtained from the open-source Flickr8K dataset that contains 8,000 images and 5 corresponding captions for each image. The first step to preprocess the data is to perform image processing to extract the input features for the images. For this, we utilized transfer learning using the VGG-16 model by the Oxford Visual Geometry Group. VGG-16 is a popular convolutional neural network model trained on the ImageNet dataset that contains over 14 million images in 1000 classes. It won the ImageNet ILSVRC competition in 2014 and is still considered as having one of the best architectures for computer vision to date (Perumanoor).   

 
![image](https://user-images.githubusercontent.com/62516990/156242132-375fb6a5-6fc4-4dda-b253-9a7b0ae69558.png)

Figure 3: CNN Architecture 

(Sugata and Yang) (https://www.researchgate.net/publication/321829624_Leaf_App_Leaf_recognition_with_deep_convolutional_neural_networks) 

A quick look at the architecture of VGG-16. VGG-16 takes the image size of 224x224 for input images. It has 13 convolutional layers with 5 max pooling layers, and 3 fully connected dense layers. The convolutional layers are 3x3 layers with stride of 1 and ReLU as the activation function. Max-pooling follow the convolutional layers and is performed over the 2×2 pixel window, with stride of 2 (Thaker). This is passed to the fully connected dense layers with the last dense layer performing the classification of 1,000 channels, and the output layer is a softmax layer (Thaker).  

Transfer learning is using a pre-trained model on a new problem where the knowledge of an already trained model is transferred to a different but closely linked problem (Sharma). In this case, the VGG-16 model has been extensively trained on image recognition, and the knowledge can be transferred to process the images for our image captioning problem. The transfer learning approach offers benefits in reduced training time by speeding up network convergence and allows for improved neural network performance even on a small dataset (Sharma).  

We begin this process by calling the pre-trained VGG-16 model in Keras. The top layers are included because we will use the fully connected layers of the model, and pretrained ‘imagenet’ is used as the weights. The inputs are set as VGG-16 model inputs, and output excluding the final dense layer and the softmax layer. This layer will be removed in our application of the model as it is the model that predicts the classification of the image, and we will only need the features (Brownlee). A dictionary is created for the encodings for the image features. Next, we start the processing step by loading each image from the image directory, resizing it into the required image size of 224x224 and 3 channels, converting image into array and then reshaping it to a 4D array to be processed by convolutional layers. This is passed to the preprocess_input function that prepares the images in our dataset to the correct format for VGG-16. Then, we collect the predicted features from the VGG-16 and store them into the encodings dictionary for each image file. These arrays are the image encodings will be used as input features for our model. 

### 3.0 CAPTION PRE-PROCESSING 

Caption pre-processing is required to convert the captions file/text in a format that it can be used to generate the model. The first step requires conversion of the caption to dictionary format and followed by caption cleaning as shown below. 

 
![image](https://user-images.githubusercontent.com/62516990/156242184-bf73ad90-8826-4c2a-839d-e20e7f631e7d.png)

Figure 4: Caption Preprocessing 

 

 

  

### 3.1 Caption Cleaning 

Caption cleaning is required to done before tokenization as we need only words input. All the operations listed below are need to be followed: 

- Lower the string  

- Removes symbols (.*?\)  

- Remove URLS  

- Remove punctuations  

- Remove next line character  

- Remove alpha numeric words 

   

### 3.2 TRAIN TEST SPLIT  

Here we split the randomly captions data into 80% train caption and 20% test captions and accordingly divided the image encodings too. Also, “sos” (Start of Statement) and “eos” (End of statement) is added to every caption with the train and test sets. This is done to set a standard pattern for the model to understand later.  

 ![image](https://user-images.githubusercontent.com/62516990/156242263-cc8f17dd-3776-4b16-b489-ca42fe78371e.png)

Figure 5: Train Test Split 

  

### 3.3 Tokenizer 

Tokenization the process of breaking a stream of textual data into words, terms, sentences, symbols, or some other meaningful elements called tokens. A lot of open-source tools are available to perform the tokenization process. A tokenizer was built to tokenize the captions into tokens to parse it in a sequence creation. Below we can see the assigned integer tokens to words within the caption of an image. 

 
![image](https://user-images.githubusercontent.com/62516990/156242302-5ead5ac9-3cc1-4ee9-bbcb-86522c0ff534.png)

Figure 6: Tokenizer 

### 3.4 Sequence Creation 

Finally, a sequence was generated using the tokenizer and the output file included a list of “x1” image encodings, “x2” input sequence for captions and y output sequence captions. 

  

  
![image](https://user-images.githubusercontent.com/62516990/156242322-9a0c7d7f-e0f5-4f94-a26b-2b930d485dda.png)

Code snippet for sequence creation 

 
![image](https://user-images.githubusercontent.com/62516990/156242347-3f4ffcc9-7c54-4a46-b8e7-7c76831e79ea.png)

Sequence Generation 


### 4.0 MODELS 

After all the data pre-processing steps, now it is time to build a model that can predict the next word of the caption for given image encodings and text sequences. For this purpose, an autoencoder architecture is selected which can be further divided into four stages: 

1. Pretrained Model to extract image features. (Encoder 1) 

2. Embedding and LSTM Layer to encode captions (Encoder 2) 

3. Add Layer to merge the output of both encoders 

4. Fully connected NN with SoftMax activation (Decoder) 

For better understanding, a high-level model architecture is built which clearly shows these stages 

 
![image](https://user-images.githubusercontent.com/62516990/156242421-8a3063ea-b794-4f24-af4a-d00677207431.png)

Figure 7: Model Architecture 

Now we will discuss each stage individually: 

1. Pretrained Model to extract image features (Encoder 1) 

A pre-trained model (VGG-16) is used to extract the features from the image. We have selected VGG-16 as it is symmetric throughout its structure and the time required to extract features is lesser than the other models such as Resnet, Inception, etc. 

 A script is implemented that uses weights of the ImageNet, and extract the features for a given image: 

 ![image](https://user-images.githubusercontent.com/62516990/156242529-db84ed9c-8783-4f7c-ada6-4f29a369afa7.png)


 

2. Embedding and LSTM Layer to encode captions (Encoder 2) 

This encoder deals with text sequences that have been created in text preprocessing. The goal is to extract a text encoding from a given text sequence. The embedding layer is the first layer that converts the given sequence (None, 37) to (None, 37, 256). It means that we have converted each token to a vector of 256 dimensions. These 256 dimensions help the next LSTM layer to understand the context of the caption. 

Next, an LSTM layer is implemented with 256 units of LSTM blocks that convert the (None, 37, 256) to (None, 256), and this represents the text encoding for a given text sequence. 

 
![image](https://user-images.githubusercontent.com/62516990/156242569-c2bc5129-1511-4707-85d9-fdce10285576.png)

Figure 8: Embeddings 

Now we have the image encoding (None, 256), and the text encodings (None, 256). An added layer acts as a bottle-neck layer and is used to merge these two encodings into a single vector of shape (None, 256). This will be fed to the decoder. 

4. Fully connected NN with SoftMax activation (Decoder) 

A dense layer with neurons same as vocab_size and SoftMax activation is used as an output layer. 

 
![image](https://user-images.githubusercontent.com/62516990/156242641-87384a64-b4e4-45a0-b992-eccb71cc097d.png)

Figure 9: NN with SoftMax 

 

### 4.1 Model Architecture: 

Now, we will merge all these stages and build a complete model from them. 

 

 

 
![image](https://user-images.githubusercontent.com/62516990/156242666-acc8851f-6764-4561-afed-0b0dadecdcb4.png)

![image](https://user-images.githubusercontent.com/62516990/156242708-8a1ee617-ea75-4adc-9649-3ce16e7e9004.png)

Figure 10: Full Model architecture 

 

 

Lastly, a greedy search algorithm is implemented to find the next most probable word. 

### 4.2 Model Evaluation: 

For evaluating our model, we used the bleu score. Bleu score (Bilingual Evaluation Understudy) is a metric that is used to measures the similarity between the machine-translated text and quality reference translations. 

The range of the bleu score is 0 to 1. Ideally, it should be one which represents the machine-translated text is the same as the reference text. 

 ![image](https://user-images.githubusercontent.com/62516990/156242745-65346337-44a5-4d3e-809d-b24992f5c788.png)


Here, we have used the unigram, bigram, trigram, and quad-gram bleu score to evaluate our model 

 ![image](https://user-images.githubusercontent.com/62516990/156242753-1d680d34-a584-44f5-8768-5a31b5eb8319.png)
 

### 4.3 Prediction on unseen images 

The model takes 2 inputs X1 and X2. X1 is the image vector which is being parsed into the model. X2 is the caption sequence. While training the model using the training data set, inputs X1 and X2 are the image vector and the caption sequence in the dataset. The model predicts the output which is the next word in the caption. The next word predicted will be appended to X2 and it will be passed to the model till the maximum length of the caption is reached or a none value is predicted by the model. 

For predicting the caption for an unseen image, the model uses the image vector X1 based on the input image. There is no caption associated with the image so the value of X2 will be the encoding for SOS(start of sentence). Using this X1 and X2 the model generates the first word of the caption as its prediction. The X2 for the next word prediction will be a encoding for SoS+ First prediction. This process will be repeated till the full caption is generated. 

The diagram below shows how the model predicts captions for unseen images: 

![image](https://user-images.githubusercontent.com/62516990/156242815-655f7646-3e8a-46f0-9b87-4d0b4c5daee4.png)

Figure 11: Prediction on unseen images 

Below shared are some of the sample predictions made by the model: 

1.
 ![image](https://user-images.githubusercontent.com/62516990/156242991-b66aa03c-899f-4890-aa4f-8aaca6dd844c.png)

- Predicted Caption: ‘sos a black dog is running through the water eos’

2.
 ![image](https://user-images.githubusercontent.com/62516990/156243178-5f77cda9-4d96-4a13-a4d5-82ae713d51f9.png)

- Predicted Caption: ‘sos a man in a black shirt is standing on a city street eos’

3. 
![image](https://user-images.githubusercontent.com/62516990/156243265-fa2f030e-fbf5-43fa-a229-f241aea193f4.png)

- Predicted Caption: ‘sos a dog is running through the grass eos’

 

### 5.0 CONCLUSION 

As a conclusion, we notice that the model performs well on training data. The scores we have gotten in model evaluation are quite favorable for our project in some aspects (like in the unigram). To improve it, we also have identified the bias in the model like you've seen in the predictions and can work on that. 

Increasing the size of our dataset as well as improving the diversity of the images and captions could help a lot also. 

For future implementations, we can think of creating an application for visually impaired people which could help them using text to speech to identify what they are not able to see.  

We can also shift the project to work with videos and improve visually impaired people's experience with movies or videos on the internet in general. 

It is also possible to merge both projects we did to create a complete solution for handicapped people, one for people who are deaf or partially deaf and one for visually impaired people. 

 

 

### 6.0 REFERENCES 

 

- Brownlee, Jason. “How to Develop a Deep Learning Photo Caption Generator from Scratch.” Machine Learning Mastery, 23 Dec. 2020, https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/.  
- Perumanoor, Tinsy John. “What Is VGG16?  -  Introduction to VGG16.” Medium, Great Learning, 23 Sept. 2021, https://medium.com/@mygreatlearning/what-is-vgg16-introduction-to-vgg16-f2d63849f615.  
- Sharma, Pranshu. “Understanding Transfer Learning for Deep Learning.” Analytics Vidhya, 30 Oct. 2021, https://www.analyticsvidhya.com/blog/2021/10/understanding-transfer-learning-for-deep-learning/.  
- Sugata, T. L. I. & Yang, C.K. ‘Leaf App: Leaf recognition with deep convolutional neural networks’ 2017, IOP Conference Series: Materials Science and Engineering. 273. 012004. 10.1088/1757-899X/273/1/012004. 
- Thaker, Tanmay. “VGG 16 Easiest Explanation.” Medium, Nerd For Tech, 8 Aug. 2021, https://medium.com/nerd-for-tech/vgg-16-easiest-explanation-12453b599526. 
- Abhijit Roy, A guide to image captioning, 9 Dec 2020, https://towardsdatascience.com/a-guide-to-image-captioning-e9fd5517f350 
- Marc Tanti, Albert Gatt, Kenneth P. Camilleri What is the Role of Recurrent Neural Networks (RNNs) in an Image Caption Generator? https://arxiv.org/pdf/1708.02043.pdf 

 
