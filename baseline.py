import os
import sys
import math
import torch
import wandb
import typer
import torch.optim as optim
from types import SimpleNamespace
from transformers import AutoTokenizer
from dataset import SubsetFineWebEdu2Loader
from transformers import GPT2Config, GPT2LMHeadModel, AdamW
from transformers import LlamaForCausalLM, LlamaConfig, LlamaTokenizer

# Argument parser for hyperparameters
def main(
    run_name: str = 'baseline',
    project_name: str = 'bistro',
    model_type: str = "llama",
    batch_size: int = 12, 
    sequence_length: int = 2048, 
    learning_rate: float = 5e-7, 
    device: str = 'cuda:1', 
    num_pages: int = 2, 
    optimizer_lr: float = 4e-4, 
    optimizer_beta1: float = 0.9, 
    optimizer_beta2: float = 0.95, 
    optimizer_weight_decay: float = 0.1,
    use_wandb: bool = False,
):
    args = SimpleNamespace(
        run_name = run_name,
        model_type = model_type,
        project_name = project_name,
        batch_size=batch_size, 
        sequence_length=sequence_length, 
        learning_rate=learning_rate, 
        device=device, 
        num_pages=num_pages, 
        optimizer_beta1=optimizer_beta1,
        optimizer_beta2=optimizer_beta2,
        optimizer_weight_decay=optimizer_weight_decay,
        use_wandb=use_wandb,
    )

    # Load the tokenizer
    batch_size = args.batch_size
    sequence_length = args.sequence_length
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained('gpt2', verbose=False)
    tokenizer.pad_token = tokenizer.eos_token

    # Create a GPT2 model from scratch.
    if args.model_type == 'gpt2':
        model = GPT2LMHeadModel( config = GPT2Config(
            output_hidden_states = False, 
            n_positions = config.sequence_length
        ))
    elif args.model_type == 'llama':
        model = LlamaForCausalLM( config = LlamaConfig(
            vocab_size = tokenizer.vocab_size,     
            hidden_size = 2040,   
            num_hidden_layers = 12,  
            num_attention_heads = 12,
            intermediate_size = 6144
        ))

    # Move model to the appropriate device
    device = args.device
    model.to(device)

    # AdamW optimizer with specified parameters
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,  # Peak learning rate
        betas=(args.optimizer_beta1, args.optimizer_beta2), # B1 and B2
        weight_decay=args.optimizer_weight_decay  # Weight decay
    )

    def train( batch ):
        """
        Perform a single training step using averaged gradient computation.
        """
        optimizer.zero_grad()

        # Shift the input ids and create labels
        input_ids = torch.tensor(batch, dtype=torch.long).to(device)
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = tokenizer.pad_token_id

        # Forward pass
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        # Backward pass
        loss.backward()
                                        
        # Apply the gradients.
        optimizer.step()
        optimizer.zero_grad()

        return loss

    # Initialize wandb if use_wandb is True
    if args.use_wandb:
        wandb.init(project=args.project_name, name = args.run_name, config=vars(args))

    model.train()
    while True:
        try:
            dataset = SubsetFineWebEdu2Loader(batch_size=batch_size, sequence_length=sequence_length, num_pages=args.num_pages, tokenizer=tokenizer)
            for idx, batch in enumerate(dataset):
                loss = train( batch )
                print(f"Loss: {loss.item()}")
                # Log metrics to wandb if use_wandb is True
                if args.use_wandb:
                    wandb.log({
                        "Loss": loss.item(),
                    })
                                            
        except Exception as e:
            import traceback
            print(f"An error occurred during training step: {e}. Continuing training...")
            traceback.print_exc()
                    
        except KeyboardInterrupt:
            print("Training interrupted. Finishing wandb run.")
            if args.use_wandb:
                wandb.finish()
            break

if __name__ == "__main__":
    typer.run(main)
