import ml_collections
from ml_collections import config_flags
from absl import app

import jax
import jax.numpy as jnp
import optax
import logging

import matplotlib.pyplot as plt

import wandb

import os
import pickle

from flax.training import checkpoints
import flax
flax.training.checkpoints.CHECKPOINT_GDA = False

# Custom imports
from src.datasets import get_dataloaders
from src.enf.model import EquivariantNeuralField
from src.enf.bi_invariants import TranslationBI, RotoTranslationBI2D
from src.enf.utils import create_coordinate_grid, initialize_latents

jax.config.update("jax_default_matmul_precision", "highest")


def save_checkpoint(checkpoint_dir, model_params, optimizer_state, epoch, global_step, best_psnr):
    """
    Saves the model parameters, optimizer state, and training metadata.
    
    Args:
        checkpoint_dir (str): Directory where the checkpoint will be saved.
        model_params (PyTree): The parameters of the model to be saved.
        optimizer_state (PyTree): The state of the optimizer.
        epoch (int): Current epoch number.
        global_step (int): Current training step.
        best_psnr (float): The best PSNR value encountered so far.
    """
    
    
    # Ensure the checkpoint directory is an absolute path
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    
    # Ensure the checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Construct checkpoint dictionary
    checkpoint_data = {
        "model_params": model_params,
        "optimizer_state": optimizer_state,
        "epoch": epoch,
        "global_step": global_step,
        "best_psnr": best_psnr,
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{global_step}.pkl")
    
    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint_data, f)
        
    print(f"Checkpoint saved at step {global_step} in {checkpoint_dir}")
    
    
def load_checkpoint(checkpoint_path):
    """
    Loads a checkpoint from a .pkl file.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    with open(checkpoint_path, "rb") as f:
        checkpoint_data = pickle.load(f)

    print(f"Checkpoint loaded from {checkpoint_path}")
    
    return (
        checkpoint_data["model_params"],
        checkpoint_data["optimizer_state"],
        checkpoint_data["epoch"],
        checkpoint_data["global_step"],
        checkpoint_data["best_psnr"],
    )

def get_config():

    # Define config
    config = ml_collections.ConfigDict()
    config.seed = 68
    config.debug = False
    config.run_name = "biobank_reconstruction"
    config.exp_name = "test"
    config.save_checkpoints = False
    config.checkpoint_path = ""
    config.bi_invariant = 'translational' 

    # Reconstruction model
    config.recon_enf = ml_collections.ConfigDict()
    
    config.recon_enf.num_hidden = 128
    config.recon_enf.num_heads = 3
    config.recon_enf.att_dim = 64
    
    config.recon_enf.num_in = 2  
    config.recon_enf.num_out = 1  
    config.recon_enf.freq_mult = (3.0, 5.0)
    config.recon_enf.k_nearest = 4
    config.recon_enf.latent_noise = True

    config.recon_enf.num_latents = 16
    config.recon_enf.latent_dim = 64
    
    config.recon_enf.even_sampling = True
    config.recon_enf.gaussian_window = True

    # Dataset config
    config.dataset = ml_collections.ConfigDict()
    config.dataset.num_workers = 0
    config.dataset.num_patients_train = 10
    config.dataset.num_patients_test = 2

    # Optimizer config
    config.optim = ml_collections.ConfigDict()
    config.optim.lr_enf = 5e-4
    config.optim.inner_lr = (2., 30., 0.) # (pose, context, window), orginally (2., 30., 0.)
    config.optim.inner_steps = 3
    config.optim.first_order_maml = False

    # Training config
    config.train = ml_collections.ConfigDict()
    config.train.batch_size = 4
    config.train.noise_scale = 1e-1  # Noise added to latents to prevent overfitting
    config.train.num_epochs_train = 10
    config.train.log_interval = 50
    config.train.debug_mode = True
    logging.getLogger().setLevel(logging.INFO)

    return config


# Set config flags
_CONFIG = config_flags.DEFINE_config_dict("config", get_config())


def plot_biobank_comparison(
    original: jnp.ndarray, 
    reconstruction: jnp.ndarray,
):
    """Plot original and reconstructed biobank images side by side.
    
    Args:
        original: Original images with shape (B, H, W, 1) or (1, H, W, 1)
        reconstruction: Reconstructed images with shape (B, H, W, 1) or (1, H, W, 1)
        poses: Optional poses to plot on the image
    """
    # Squeeze out batch and channel dimensions to get 2D arrays (H, W)
    original = jnp.squeeze(original)
    reconstruction = jnp.squeeze(reconstruction)
    
    fig, axes = plt.subplots(1, 2, figsize=(4, 2))
    fig.suptitle('Original vs Reconstruction')
    
    # Clip to prevent warnings
    original = jnp.clip(original, 0, 1)
    reconstruction = jnp.clip(reconstruction, 0, 1)

    # Plot original
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original')

    # Plot reconstructed
    axes[1].imshow(reconstruction, cmap='gray')
    axes[1].set_title('Reconstruction')

    # Remove axes
    for ax in axes.flat:
        ax.axis('off')
    
    plt.tight_layout()
    return fig


def main(_):

    # Get config
    config = _CONFIG.value

    # Initialize wandb
    run = wandb.init(project=config.exp_name, config=config.to_dict(), mode="online" if not config.debug else "dryrun", name=config.run_name)

    # Load dataset, get sample image, create corresponding coordinates
    train_dloader, test_dloader = get_dataloaders('biobank_endpoints', config.train.batch_size, config.dataset.num_workers, seed=config.seed, debug_mode=config.train.debug_mode)
    sample_img, _ = next(iter(train_dloader))
    img_shape = sample_img.shape[1:]  

    # Random key
    key = jax.random.PRNGKey(55)

    # Create coordinate grid for this dataset
    x = create_coordinate_grid(batch_size=config.train.batch_size, img_shape=img_shape)
    
    if config.bi_invariant == 'translational':
        bi_invariant = TranslationBI()
    elif config.bi_invariant == 'rotational':
        bi_invariant = RotoTranslationBI2D()
        

    # Define the reconstruction model
    recon_enf = EquivariantNeuralField(
        num_hidden=config.recon_enf.num_hidden,
        att_dim=config.recon_enf.att_dim,
        num_heads=config.recon_enf.num_heads,
        num_out=config.recon_enf.num_out,
        emb_freq=config.recon_enf.freq_mult,
        nearest_k=config.recon_enf.k_nearest,
        bi_invariant=TranslationBI(),
        gaussian_window=config.recon_enf.gaussian_window,
    )

    # Create dummy latents for model init
    key, subkey = jax.random.split(key)
    temp_z = initialize_latents(
        batch_size=1,  
        num_latents=config.recon_enf.num_latents,
        latent_dim=config.recon_enf.latent_dim,
        data_dim=config.recon_enf.num_in,
        bi_invariant_cls=TranslationBI,
        key=subkey,
        noise_scale=config.train.noise_scale,
        even_sampling=config.recon_enf.even_sampling,
        latent_noise=config.recon_enf.latent_noise,
    )

    # Init the model
    recon_enf_params = recon_enf.init(key, x, *temp_z)

    # Define optimizer for the ENF backbone
    enf_opt = optax.adam(learning_rate=config.optim.lr_enf)
    recon_enf_opt_state = enf_opt.init(recon_enf_params)

    @jax.jit
    def recon_inner_loop(enf_params, coords, img, key):
        z = initialize_latents(
            batch_size=config.train.batch_size,
            num_latents=config.recon_enf.num_latents,
            latent_dim=config.recon_enf.latent_dim,
            data_dim=config.recon_enf.num_in,
            bi_invariant_cls=TranslationBI,
            key=key,
            noise_scale=config.train.noise_scale,
            latent_noise=config.recon_enf.latent_noise,
        )

        def mse_loss(z):
            out = recon_enf.apply(enf_params, coords, *z)
            return jnp.sum(jnp.mean((out - img) ** 2, axis=(1, 2)), axis=0)
        
        def psnr(z):
            out = recon_enf.apply(enf_params, coords, *z)
            mse = jnp.mean((img - out) ** 2, axis=1) 
            
            # TODO: Check if max_pixel_value is correct
            max_pixel_value = 1.0 
            psnr = 20 * jnp.log10(max_pixel_value / jnp.sqrt(mse))
            
            return jnp.mean(psnr)

        def inner_step(z, _):
            _, grads = jax.value_and_grad(mse_loss)(z)
            # Gradient descent update
            z = jax.tree.map(lambda z, grad, lr: z - lr * grad, z, grads, config.optim.inner_lr)
            return z, None
        
        # Perform inner loop optimization
        z, _ = jax.lax.scan(inner_step, z, None, length=config.optim.inner_steps)
        
        # Stop gradient if first order MAML
        if config.optim.first_order_maml:
            z = jax.lax.stop_gradient(z)
            
            
        psnr_value = psnr(z) 
        psnr_value = jax.lax.stop_gradient(psnr_value)
        
        return mse_loss(z), (z, psnr_value)

    @jax.jit
    def recon_outer_step(coords, img, enf_params, enf_opt_state, key):
        # Perform inner loop optimization
        key, subkey = jax.random.split(key)
        (loss, (z, psnr_value)), grads = jax.value_and_grad(recon_inner_loop, has_aux=True)(enf_params, coords, img, key)

        # Update the ENF backbone
        enf_grads, enf_opt_state = enf_opt.update(grads, enf_opt_state)
        enf_params = optax.apply_updates(enf_params, enf_grads)

        # Sample new key
        return (loss, z, psnr_value), enf_params, enf_opt_state, subkey
    
    
    def evaluate_test_set(enf_params, test_dloader, key):
        """Evaluate model on the entire test set."""
        psnrs = []
        mses = []
        context_stds = []
        
        # Randomly select which batch to return for visualization
        key, subkey = jax.random.split(key)
        batch_idx_to_return = jax.random.randint(subkey, (), 0, len(test_dloader))
        
        # Store randomly selected batch for visualization
        vis_batch_original = None
        vis_batch_recon = None
        
        for i, (img, _) in enumerate(test_dloader):
            # Flatten input
            y = jnp.reshape(img, (img.shape[0], -1, img.shape[-1]))
            
            # Get new key
            key, subkey = jax.random.split(key)
            
            mse, (z, psnr_value) = recon_inner_loop(enf_params, x, y, key)
            
            
            # Calculate std of context vectors (z[1] contains context vectors)
            context_std = jnp.mean(jnp.std(z[1], axis=1))  # Mean std over all dimensions
            context_stds.append(context_std)
            
            psnrs.append(psnr_value)
            mses.append(mse)
            
            # Store reconstruction for randomly selected batch
            if i == batch_idx_to_return:
                vis_batch_original = img
                vis_batch_recon = recon_enf.apply(enf_params, x, *z).reshape(img.shape)
             
        # Calculate averages
        avg_psnr = jnp.mean(jnp.array(psnrs))
        avg_mse = jnp.mean(jnp.array(mses))
        avg_context_std = jnp.mean(jnp.array(context_stds))
        
        # Return randomly selected batch for visualization and the averages
        return avg_psnr, avg_mse, vis_batch_original, vis_batch_recon, avg_context_std
    
    if os.path.exists(config.checkpoint_path):
        logging.info(f"\033[93mResuming training from checkpoint: {config.checkpoint_path}\033[0m")
        recon_enf_params, recon_enf_opt_state, start_epoch, glob_step, best_psnr = load_checkpoint(config.checkpoint_path)
        
        logging.info(f"Starting epoch: {start_epoch}")
        logging.info(f"Global step: {glob_step}")
        logging.info(f"Best PSNR: {best_psnr}")
    else:
        logging.info("\033[91mNo checkpoint found. Starting training from scratch.\033[0m")
    
 
    # Pretraining loop for fitting the ENF backbone
    best_psnr = float('-inf')
    glob_step = 0
    for epoch in range(config.train.num_epochs_train):
        epoch_loss = []
        epoch_psnr = []
        for i, (img, _) in enumerate(train_dloader):
            y = jnp.reshape(img, (img.shape[0], -1, img.shape[-1]))

            # Perform outer loop optimization
            (loss, z, psnr_value), recon_enf_params, recon_enf_opt_state, key = recon_outer_step(
                x, y, recon_enf_params, recon_enf_opt_state, key)

            epoch_loss.append(loss)
            epoch_psnr.append(psnr_value)
            glob_step += 1

            if glob_step % config.train.log_interval == 0:
                # Reconstruct and plot the first image in the batch
                
                test_psnr, test_mse, img, img_r, context_std = evaluate_test_set(recon_enf_params, test_dloader, key)
                
                if test_psnr > best_psnr:
                    best_psnr = test_psnr
                    if config.save_checkpoints:
                        save_checkpoint(f"model_checkpoints/{config.exp_name}/{config.run_name}", recon_enf_params, recon_enf_opt_state, epoch, glob_step, best_psnr)
                    
                # getting random int between 0 and batch size
                random_idx = jax.random.randint(key, (1,), 0, config.train.batch_size)
                fig = plot_biobank_comparison(img[random_idx], img_r[random_idx])
                
                wandb.log({
                    "recon-mse": sum(epoch_loss) / len(epoch_loss), 
                    "test-mse": test_mse, 
                    "test-psnr": test_psnr,
                    "context-std": context_std,
                    "reconstruction": fig,
                    "epoch": epoch
                }, step=glob_step)
                plt.close('all')
                logging.info(f"RECON ep {epoch} / step {glob_step} || mse: {sum(epoch_loss[-10:]) / len(epoch_loss[-10:])} || test-mse: {test_mse} || test-psnr: {test_psnr} || context-std: {context_std}")

   
    run.finish()


if __name__ == "__main__":
    app.run(main)
