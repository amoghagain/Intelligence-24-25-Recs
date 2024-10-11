IMAGE ENHANCEMENT TASK:

This task involves mainly involves enhancement of images using 3 methods mainly

1) VAE(Variational Autoencoder)
1) GAN (Generative Adversial Networks)
1) Diffusion Model

These are compared on basis of:

·  **PSNR**: Measures the peak error; higher values are better.

·  **SSIM**: Measures structural similarity; values closer to 1 are better.

·  **MSE**: Measures average squared error; lower values are better.

**1. Peak Signal-to-Noise Ratio (PSNR)**

- **Definition**: PSNR is a measure of the peak error between two images. It is calculated using the logarithm of the ratio between the maximum possible power of a signal (image) and the power of the noise that affects the fidelity of its representation.

**2. Structural Similarity Index (SSIM)**

- **Definition**: SSIM is a perceptual metric that quantifies the similarity between two images. It considers changes in structural information, luminance, and contrast. Unlike PSNR, which treats images as a set of pixel values, SSIM is designed to model human visual perception.

**3. Mean Squared Error (MSE)**

- **Definition**: MSE measures the average squared difference between the pixel values of the original and generated images. It quantifies the error introduced by distortion or noise.

How do these models work?

Subtask\_1 : Variational Autoencoders (VAEs) are a type of generative model that can be used for various tasks, including image enhancement. Here's how VAEs work and how they can be applied to enhance images:

**Overview of VAEs**

1. **Architecture**:
   1. A VAE consists of two main components: the **encoder** and the **decoder**.
      1. **Encoder**: Maps input images to a latent space representation (latent variables).
      1. **Decoder**: Reconstructs images from the latent space representation.
1. **Latent Space**:
   1. The encoder learns to compress the input images into a lower-dimensional latent space, characterized by a mean and a variance. Instead of encoding a single point in the latent space, the encoder outputs parameters for a probability distribution (typically Gaussian).
1. **Reparameterization Trick**:
   1. To allow for backpropagation during training, VAEs use the reparameterization trick. Instead of sampling directly from the latent distribution, the model samples from a standard normal distribution and transforms it using the learned mean and variance. This enables gradient flow through the sampling process.
1. **Loss Function**:
   1. The VAE loss function consists of two parts:
      1. **Reconstruction Loss**: Measures how well the decoder reconstructs the input images from the latent representation. This is typically calculated using pixel-wise losses (e.g., Mean Squared Error).
      1. **KL Divergence**: Measures how closely the learned latent distribution approximates the prior distribution (usually a standard normal distribution). This term regularizes the latent space, encouraging it to be continuous and smooth.

**How VAEs Work for Image Enhancement**

1. **Training**:
   1. VAEs are trained on a dataset of images (e.g., raw images) where the model learns to reconstruct these images from the latent space. The encoder captures the essential features of the images, while the decoder aims to reproduce high-quality outputs.
1. **Latent Space Representation**:
   1. During training, the model learns a representation of the data that captures meaningful variations and structures. This representation can be used to manipulate or enhance images.
1. **Image Enhancement Process**:
   1. **Noise Reduction**:
      1. For image enhancement, especially in tasks like denoising, noisy images are fed into the VAE. The encoder maps the noisy images to the latent space, and the decoder reconstructs the images based on this latent representation. Since the VAE has learned to reconstruct clean images, it can effectively reduce noise in the input images.
   1. **Image Generation**:
      1. The trained VAE can generate new images by sampling from the latent space. By manipulating points in the latent space, you can create enhanced images that retain important features from the original data while improving quality.
   1. **Feature Manipulation**:
      1. You can also explore the latent space to enhance certain features or aspects of the images. For example, moving in specific directions in the latent space may increase brightness, sharpness, or other desired attributes.

Subtask\_2\_3: Generative Adversarial Networks (GANs) are a class of deep learning models used for generative tasks, such as image generation, image enhancement, and more. Here's an overview of how GANs work:

**Overview of GANs**

1. **Architecture**:
   1. A GAN consists of two main components: the **generator** and the **discriminator**.
      1. **Generator**: Creates fake images from random noise (latent vectors) to imitate real data.
      1. **Discriminator**: Evaluates images and determines whether they are real (from the dataset) or fake (generated by the generator).
1. **Adversarial Training**:
   1. The generator and discriminator are trained simultaneously in a competitive manner:
      1. The generator aims to produce images that are indistinguishable from real images.
      1. The discriminator tries to correctly classify real and fake images.
   1. This adversarial setup drives both networks to improve over time.

**How GANs Work**

1. **Generator Network**:
   1. The generator takes a random noise vector (sampled from a probability distribution, often a Gaussian) and processes it through multiple layers (e.g., convolutional layers) to produce an image.
   1. The objective of the generator is to maximize the probability of the discriminator making a mistake (i.e., classifying a generated image as real).
1. **Discriminator Network**:
   1. The discriminator takes both real images from the training dataset and fake images generated by the generator.
   1. It outputs a probability score indicating whether the input image is real or fake.
   1. The objective of the discriminator is to correctly classify real images as real and fake images as fake.
1. **Training Process**:
   1. **Step 1**: Sample a batch of real images from the dataset.
   1. **Step 2**: Generate a batch of fake images using the generator.
   1. **Step 3**: Train the discriminator using both real and fake images. The discriminator learns to distinguish between them.
   1. **Step 4**: Update the generator's parameters based on the discriminator's performance. This is done by minimizing the discriminator’s ability to classify fake images correctly.
   1. **Step 5**: Repeat steps 1 to 4 until the generator produces high-quality images, and the discriminator can no longer easily distinguish between real and fake images.
1. **Loss Functions**:
   1. The loss functions for the generator and discriminator are based on their respective objectives:
      1. **Discriminator Loss**: Measures how well the discriminator can classify real and fake images. This loss typically involves binary cross-entropy.
      1. **Generator Loss**: Measures how well the generator fools the discriminator. The generator's loss is based on the discriminator's output for generated images, encouraging it to produce better-quality images over time.

**Image Generation and Enhancement Process**

1. **Sampling**:
   1. After training, to generate new images, you can sample random noise vectors from the latent space and feed them into the generator. The generator will produce corresponding images.
1. **Image Enhancement**:
   1. GANs can also be used for image enhancement tasks, such as super-resolution, inpainting, and denoising. In these cases, specialized architectures (e.g., Super Resolution GANs) modify the standard GAN framework to optimize for specific enhancement objectives.
### **Subtask\_4: What Are Diffusion Models?**
Diffusion models are a type of generative model used to create new images. They work by simulating a process that gradually transforms images into random noise and then learns to reverse this process to generate new images.

**How Diffusion Models Work**

1. **Adding Noise**:
   1. Imagine you have a clear image. In the first step, the model starts adding random noise to this image. This process is repeated multiple times, gradually making the image more and more noisy until it looks like a completely random pattern. By the end of this process, the original image is no longer recognizable.
1. **Learning to Denoise**:
   1. The model learns how to remove this noise step by step. During training, it sees both the noisy images and the original images. The goal is to teach the model how to take a noisy image and predict what the original image looked like before the noise was added.
1. **Training the Model**:
   1. To train the model, it looks at many pairs of original and noisy images. It learns to recognize patterns in the noise and how to reverse these patterns to recover the original images. This involves figuring out what the noise looks like and how to remove it effectively.
1. **Generating New Images**:
   1. Once the model is trained, you can use it to create new images. You start with a random noise image and then apply the learned process to gradually reduce the noise, turning it into a new, coherent image. This process is done in several steps, where at each step, the model improves the image by reducing the noise.

So based on my models, the best are :
GAN, Diffusion Model and then VAE’s for same dataset.

Generally diffusion models perform better than GAN’s
This might be due to not so perfect model and architecture.

The above result is based on PSNR, MSE and SSIM scores

