# The MIT License (MIT)
# Copyright © 2024 Chakana.tech

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import io
import os
import copy
import math
import time
import boto3
import torch
import wandb
import typer
import random
import argparse
import tempfile
import bittensor as bt
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from dotenv import dotenv_values
from types import SimpleNamespace
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Tuple

# Import common tooling.
from common import upload_model, get_latest_metadata, download_model, hash_model
from dataset import SubsetFineWebEdu2Loader

# Instantiate my S3 client.
env_config = {**dotenv_values(".env"), **os.environ}
AWS_ACCESS_KEY_ID = env_config.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = env_config.get('AWS_SECRET_ACCESS_KEY')
CLIENT: boto3.client = boto3.client(
    's3',
    region_name='us-east-1',
    aws_access_key_id = AWS_ACCESS_KEY_ID,
    aws_secret_access_key = AWS_SECRET_ACCESS_KEY
)

# Main function.
def main( config ):
    print('\n', '-' * 40, 'Config', '-' * 40,)
    print ( config )
    
    # Init Bittensor objects.
    wallet = bt.wallet( config = config )
    subtensor = bt.subtensor( config = config )
    metagraph = subtensor.metagraph( netuid = config.netuid )
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        raise ValueError(f'Wallet {wallet} is not registered on subnet: {metagraph.netuid}')
    my_uid = metagraph.hotkeys.index( wallet.hotkey.ss58_address )
    print('\n', '-' * 40, 'Objects', '-' * 40,)
    print ( f'Wallet: {wallet}\nSubtensor: {subtensor}\nMetagraph: {metagraph}\nUID: {my_uid}' )
    
    # Assert the chain commitment.
    try:
        if config.bucket != subtensor.get_commitment(config.netuid, my_uid):
            raise ValueError(f'Chain commitment does not match: {config.bucket}')
    except Exception: 
        subtensor.commit( wallet, config.netuid, config.bucket)
    print('Bucket:', config.bucket , '\n')
                    
    # Init weights and biases
    run = None
    if config.use_wandb:
        name = f'Miner-{wallet.hotkey.ss58_address[:5]}' if not config.baseline else 'Baseline'
        run = wandb.init( project='bistro', resume = 'allow', name = name, config = config )
    
    # Main training loop.
    steps = 0
    master = None
    n_failed_sync = 0
    n_success_sync = 0
    history = [] # history of previous delta.s
    while True:
        
        try:
            # Sync chain state.
            master_uid = int( metagraph.S.argmax() )
            subtensor = bt.subtensor( config = config )
            metagraph = subtensor.metagraph( netuid = config.netuid )
            master_meta = get_latest_metadata( master_uid, metagraph, subtensor, CLIENT = CLIENT )
            if master_meta == None:
                print ('\tWaiting for a master model to be uploaded ... ')
                time.sleep(5)
                continue
            steps += 1 # New step master is uploaded.
            if config.use_wandb: wandb.log({ "step": steps, "block": subtensor.block, "master": master_uid} )
            print('\n', '-' * 40, f'Step: {steps}', '-' * 40)
            print ( f'Block: {subtensor.block}, Master: { master_uid } Hash: { master_meta.model_hash }' )
            
            # If we are not in sync with master, download the state.
            if hash_model( master ) != master_meta.model_hash:
                # Fully resync the state.
                print ('Failed to sync master state using deltas.')
                master = download_model( metadata = master_meta, device = 'cpu', CLIENT = CLIENT )
                tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained( master_meta.tokenizer_name, verbose=False, clean_up_tokenization_spaces=True )
                tokenizer.pad_token = tokenizer.eos_token    
                master_hash = hash_model( master )
                optimizer = optim.AdamW(
                    master.parameters(),
                    lr = config.learning_rate,  # Peak learning rate
                    betas = ( config.optimizer_beta1, config.optimizer_beta2 ), # B1 and B2
                    weight_decay = config.optimizer_weight_decay  # Weight decay
                )
                master.to( config.device )
                master.train()
                n_failed_sync += 1
            else:
                print ('Successfully synced master state using deltas.')
                n_success_sync += 1
            if config.use_wandb: wandb.log({ "n_failed_sync": n_failed_sync, "n_success_sync": n_success_sync } )


            # Record previous state.
            checkpoint = copy.deepcopy( master ).to('cpu') # For recording last master state.
            
            # Train until master is updated.
            while True:
                
                # Pull random pages from the evaluation window.
                # These are the pages the miner delta will be evaluated on.
                eval_pages: Tuple[ str, int, str ] = SubsetFineWebEdu2Loader.next_pages( offset = subtensor.block, n_pages = config.eval_window, seed = my_uid )            
                dataset = SubsetFineWebEdu2Loader(
                    batch_size = config.batch_size,
                    sequence_length = master_meta.sequence_length,
                    pages_info = random.sample( eval_pages, min( config.eval_window, config.num_pages_per_upload ) ),
                    tokenizer = tokenizer
                )
                
                # Train...
                for batch in dataset:
                    
                    # Forward pass
                    input_ids = torch.tensor(batch, dtype=torch.long).to(config.device)
                    labels = input_ids.clone()
                    labels = torch.where( labels == tokenizer.pad_token_id, -100, labels )
                    outputs = master( input_ids = input_ids, labels = labels )
                    
                    # Accumulate gradients.
                    outputs.loss.backward()
                    print ( 'loss', outputs.loss.item() )
                    if config.use_wandb: wandb.log({ "loss": outputs.loss.item() } )

                    # Step the optimizer.
                    optimizer.step()
                    optimizer.zero_grad()
                    break
                    
                # Baseline just keeps training.
                if config.baseline: continue
                    
                # Compute the delta between the current master model and the checkpoint model in-place
                delta = copy.deepcopy( master ).to('cpu')
                for (name, param), (_, checkpoint_param) in zip(delta.named_parameters(), checkpoint.named_parameters()):
                    param.data.sub_(checkpoint_param.data)
                    
                # Upload the delta to S3 and check state.
                history.append( upload_model(
                    wallet = wallet,
                    model = delta,
                    block = int(time.time()),
                    extras = { 'master_hash': master_meta.model_hash },
                    bucket = config.bucket,
                    CLIENT = CLIENT,
                    use_compression = config.use_compression,
                    compression_percent = config.compression_percent,
                ))
                if len(history) > config.history_size:
                    old_delta = history.pop(0)
                    print (f'Deleting old delta: {old_delta.filename}, still holding: {len(history)} files.')
                    CLIENT.delete_object( Bucket = config.bucket, Key = old_delta.filename )
                    CLIENT.delete_object( Bucket = config.bucket, Key = old_delta.metadata_filename )

                # Check if the master has changed.
                latest_master_meta = get_latest_metadata( master_uid, metagraph, subtensor, CLIENT = CLIENT )
                print ('latest_master_meta', latest_master_meta )
                if master_hash != latest_master_meta.model_hash:
                    print ('A new master has been uploaded, break the loop.')
                    break 

            # Download the deltas and attain the new master.
            print ('Sync the new state.')
            print ( 'Deltas', latest_master_meta.deltas )
            for delta_meta in latest_master_meta.deltas:
                as_namespace = SimpleNamespace( **delta_meta )
                print ('applying delta: ', {as_namespace.filename})
                delta = download_model( metadata = as_namespace, device = 'cpu', CLIENT = CLIENT )
                for (name, checkpoint_param), (_, delta_param) in zip( checkpoint.named_parameters(), delta.named_parameters() ):
                    delta_update = delta_param.data.to( checkpoint.device )
                    if torch.isnan( delta_update ).any(): delta_update[ torch.isnan(delta_update) ] = 0  # Set NaNs to 0
                    checkpoint_param.data.add_( delta_update / len( latest_master_meta.deltas ) ) 
                    
            # Then reset the master to the new updated checkpoint + delta.
            del master
            del delta
            torch.cuda.empty_cache()
            master = copy.deepcopy( checkpoint )
            master_hash = hash_model( master )
            master.to(config.device)
            master.train()
            optimizer = optim.AdamW(
                master.parameters(),
                lr = config.learning_rate,  # Peak learning rate
                betas = ( config.optimizer_beta1, config.optimizer_beta2 ), # B1 and B2
                weight_decay = config.optimizer_weight_decay  # Weight decay
            )

        # Handle keyboard interrupts, stops training gracefully.
        except (KeyboardInterrupt, SystemExit):
            if config.use_wandb and run != None: 
                api = wandb.Api()
                api_run = api.run(f"{run.entity}/{run.project}/{run.id}")
                api_run.delete()
            for el in history[:-1]:
                print (f'Deleting: {el.filename}')
                CLIENT.delete_object( Bucket = config.bucket, Key = el.filename )
                CLIENT.delete_object( Bucket = config.bucket, Key = el.metadata_filename )
            break
        
        # Handle unknown exceptions, continue training after 5 seconds.
        except Exception as e:
            print (f"Error: {e}")
            time.sleep(5)
            continue

# Main function.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Miner script')
    parser.add_argument('--name', type=str, default=None, help='Optional miner name')
    parser.add_argument('--netuid', type=int, default=212, help='Bittensor network uid.')
    parser.add_argument('--bucket', type=str, default='decis', help='S3 bucket name')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--history_size', type=int, default=4, help='Number of previous gradients to maintain.')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for the optimizer')
    parser.add_argument('--optimizer_beta1', type=float, default=0.9, help='Beta1 for the optimizer')
    parser.add_argument('--optimizer_beta2', type=float, default=0.95, help='Beta2 for the optimizer')
    parser.add_argument('--optimizer_weight_decay', type=float, default=0.1, help='Weight decay for the optimizer')
    parser.add_argument('--num_pages_per_upload', type=int, default=1, help='Number of pages per delta upload')
    parser.add_argument('--eval_window', type=int, default=5, help='Number of pages to load')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights and Biases for logging')
    parser.add_argument('--baseline', action='store_true', help='Baseline addition.')
    parser.add_argument('--use_compression', action='store_true', help='If the delta should use compression')
    parser.add_argument('--compression_percent', type=float, default=0.9, help='Compression percentage default 90%.')
    bt.wallet.add_args( parser )
    bt.subtensor.add_args( parser )
    config = bt.config( parser )   
    config.subtensor.chain_endpoint = 'wss://test.finney.opentensor.ai:443/' # Fix this value.
    main( config ) 