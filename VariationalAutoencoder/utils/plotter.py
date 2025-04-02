import torch
from matplotlib import pyplot as plt
import os

def plot_vae_generations(model, device, num_images=5, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)

    z_samples = torch.randn(num_images, model.latent).to(device)
    images = []

    with torch.no_grad():
        for sample in z_samples:
            z = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)
            generated_image = model.decode(z)
            generated_image = torch.nn.functional.sigmoid(generated_image).view(28, 28).clone().detach().cpu()


            images.append(generated_image)

    fig, axs = plt.subplots(1, num_images, figsize=(15,3))
    for i in range(num_images):
        axs[i].imshow(images[i].numpy().squeeze(), cmap="gray")
        axs[i].axis('off')

    output_path = os.path.join(output_dir, f"generated_images.png")
    plt.savefig(output_path)
    plt.close()