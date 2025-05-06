# train_fsdp.py
import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, enable_wrap, wrap
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp import BackwardPrefetch
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp import StateDictType
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from torch.optim import AdamW 
from datasets import load_dataset
from transformer_lens import HookedTransformer
import typeguard

def main():
    # Initialize distributed training
    if 'SLURM_PROCID' in os.environ:
        # We're running under SLURM
        rank = int(os.environ['SLURM_PROCID'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        master_addr = os.environ['MASTER_ADDR']
        master_port = os.environ['MASTER_PORT']
    else:
        # Fallback for non-SLURM environment
        rank = int(os.environ.get('RANK', 0))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        master_addr = os.environ.get('MASTER_ADDR', 'localhost')
        master_port = os.environ.get('MASTER_PORT', '29500')

    # Set up the process group
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=rank
    )
    
    # Set the device
    torch.cuda.set_device(local_rank)

    # Create checkpoint directory
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Saving checkpoints to: {checkpoint_dir}")

    # Only print on rank 0
    if rank == 0:
        print(f"Initialized process group: rank={rank}, world_size={world_size}, local_rank={local_rank}")

    model = AutoModelForCausalLM.from_pretrained(
        "gpt2", torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    
    # Configure mixed precision
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16
    )
    
    # Configure FSDP
    model = FSDP(
        model,
        auto_wrap_policy=size_based_auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=torch.cuda.current_device(),
    )

    tok = AutoTokenizer.from_pretrained("gpt2")
    # Set up padding token
    tok.pad_token = tok.eos_token
    data = load_dataset("roneneldan/TinyStories", split="train[:1%]")

    def collate(batch):
        texts = [item['text'] for item in batch]
        toks = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        return {"input_ids": toks["input_ids"], "labels": toks["input_ids"].clone()}

    loader = torch.utils.data.DataLoader(
        data, batch_size=2, shuffle=True, collate_fn=collate
    )

    optim = AdamW(model.parameters(), lr=2e-4)
    sched = get_scheduler("cosine", optim, num_warmup_steps=10, num_training_steps=200)

    # --- bonus: capture block-4 activations for later mech-interp ----------
    hook_handle = None
    hooked_transformer = None
    if rank == 0:  # avoid all ranks writing to disk
        act_dir = os.path.join(os.path.dirname(__file__), "activations")
        os.makedirs(act_dir, exist_ok=True)
        print(f"Saving activations to: {act_dir}")
        
        # Initialize HookedTransformer
        hooked_transformer = HookedTransformer.from_pretrained("gpt2")
        hooked_transformer.eval()  # Set to eval mode
        
        # Define hook function
        def save_act(module, input_tensor, output_tensor):
            # Save activation tensor with step number
            save_path = os.path.join(act_dir, f"act_block4_output_step_{step}.pt")
            print(f"Saving activation to: {save_path}")
            torch.save({
                'output': output_tensor.detach().cpu(),
                'input': input_tensor[0].detach().cpu() if isinstance(input_tensor, tuple) else input_tensor.detach().cpu(),
                'step': step
            }, save_path)
            return output_tensor
        
        # Register hook
        hook_handle = hooked_transformer.blocks[4].register_forward_hook(save_act)
        print("Registered activation hook on block 4")

    model.train()
    for step, batch in enumerate(loader):
        batch = {k: v.cuda() for k, v in batch.items()}
        loss = model(**batch).loss
        loss.backward()
        optim.step(); sched.step(); optim.zero_grad()
        
        if step % 20 == 0:
            if rank == 0:
                print(f"[{step}] loss={loss.item():.3f}")
                # Save checkpoint
                with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
                    state_dict = model.state_dict()
                    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pt")
                    torch.save({
                        'model_state_dict': state_dict,
                        'optimizer_state_dict': optim.state_dict(),
                        'scheduler_state_dict': sched.state_dict(),
                        'step': step,
                        'loss': loss.item(),
                    }, checkpoint_path)
                    print(f"Saved checkpoint to: {checkpoint_path}")
                # experimental: token saving
                tok_path = os.path.join(act_dir, f"input_ids_step_{step}.pt")
                torch.save(batch["input_ids"].detach().cpu(), tok_path)
                # Run hooked transformer on current batch
                if hooked_transformer is not None:
                    print(f"Running hooked transformer forward pass for activation capture at step {step}")
                    with torch.no_grad():
                        hooked_transformer(batch["input_ids"].cpu())

        if step == 199: break

    if hook_handle is not None:
        hook_handle.remove()
        print("Removed activation hook")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
