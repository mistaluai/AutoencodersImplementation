import random
import matplotlib.pyplot as plt
import os
import torch

def plot_random_reconstructions(model, dataloader, device, num_images=5, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)

    # Collect all images from the validation DataLoader
    all_images = []
    for batch_images, _ in dataloader:
        all_images.extend(batch_images)
        if len(all_images) >= num_images:
            break

    # Select random images
    random_images = random.sample(all_images, num_images)

    # Reconstruct and save plots
    for i, image in enumerate(random_images):
        # Move the image to the device
        image = image.view(1, -1).to(device, dtype=torch.float32)

        # Reconstruct the image
        with torch.no_grad():
            reconstructed_image = model(image).cpu().view(28, 28)

        # Reshape the original image for plotting
        input_image = image.cpu().view(28, 28)

        # Plot the original and reconstructed image
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.title("Input Image")
        plt.imshow(input_image, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("Reconstructed Image")
        plt.imshow(reconstructed_image, cmap='gray')
        plt.axis('off')

        # Save the plot
        output_path = os.path.join(output_dir, f"reconstruction_{i + 1}.png")
        plt.savefig(output_path)
        plt.close()