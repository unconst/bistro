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

import os
import copy
import math
import time
import boto3
import torch
import wandb
import signal
import random
import argparse
import traceback
import numpy as np
import bittensor as bt
from tqdm import tqdm
from collections import deque
from dotenv import dotenv_values
from types import SimpleNamespace
from transformers import AutoTokenizer
from transformers import GPT2Config, GPT2LMHeadModel
from transformers import LlamaForCausalLM, LlamaConfig, LlamaTokenizer
from typing import Dict, Optional

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

# For gracefull exits.
def handle_exit_signal(signum, frame, config):
    if config.use_wandb:
        wandb.finish()
        api = wandb.Api()
        run = api.run(f"{wandb.run.entity}/{wandb.run.project}/{wandb.run.id}")
        run.delete()
    exit(0)
signal.signal(signal.SIGTERM, handle_exit_signal)

# Main function.
def main( config ):
    print ( config )
    
    # Init Bittensor objects.
    wallet = bt.wallet( config = config )
    subtensor = bt.subtensor( config = config )
    metagraph = subtensor.metagraph( netuid = config.netuid )
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        raise ValueError(f'Wallet {wallet} is not registered on subnet: {metagraph.netuid}')
    my_uid = metagraph.hotkeys.index( wallet.hotkey.ss58_address )
    print ( f'Wallet: {wallet}\nSubtensor: {subtensor}\nMetagraph: {metagraph}\nUID: {my_uid}' )
    
    # Init weights and biases
    if config.use_wandb:
        name = f'Validator-{wallet.hotkey.ss58_address[:5]}'
        wandb.init(project='bistro', name = name, config = config )
        
    # Remember delta for later removal.
    steps = 0
    master = None
    weights = torch.zeros( (metagraph.n), dtype=torch.float32)
    while True:
        try:
        
            # Sync chain state.
            master_uid = 0 #int( metagraph.S.argmax() )
            subtensor = bt.subtensor( config = config )
            metagraph = subtensor.metagraph( netuid = config.netuid )
            master_meta = get_latest_metadata( master_uid, metagraph, subtensor, CLIENT = CLIENT )
            if master_meta == None:
                print ('\tWaiting for a master model to be uploaded ... ')
                time.sleep(5)
                continue
            steps += 1
            if config.use_wandb: wandb.log({ "step": steps, "block": subtensor.block, "master": master_uid} )
            print('-' * 80)
            print ( f'Step: { steps }, Block: { subtensor.block }, Master: { master_uid } Hash: { master_meta.model_hash }' )

                        
            # If we are not in sync with master, download the state.
            if hash_model( master ) != master_meta.model_hash:
                master = download_model( metadata = master_meta, device = config.device, CLIENT = CLIENT )
                tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained( master_meta.tokenizer_name, verbose=False, clean_up_tokenization_spaces=True )
                tokenizer.pad_token = tokenizer.eos_token    
                master_hash = hash_model( master )
                master.eval()
                
            # Eval until the master has updated.
            while True:
                
                # Get the next miner to eval.
                miner_uid = random.choice( metagraph.uids )
                if miner_uid == master_uid: continue # Dont eval the master.
                
                # Get the miner metadata
                miner_meta = get_latest_metadata( miner_uid, metagraph, subtensor, CLIENT = CLIENT )
                if miner_meta == None or miner_meta.master_hash != master_hash:
                    # Miner meta is non existent or out of sync with the master.
                    continue

                # Download the delta.
                miner_delta = download_model( metadata = miner_meta, device = 'cpu', CLIENT = CLIENT )
                if miner_delta == None:
                    # Failed to download the delta.
                    continue
        
                # Pull random pages from the evaluation window.
                eval_pages: Tuple[ str, int, str ] = SubsetFineWebEdu2Loader.next_pages( offset = subtensor.block, n_pages = config.eval_window, seed = miner_uid )
                dataset = SubsetFineWebEdu2Loader(
                    batch_size = config.batch_size,
                    sequence_length = master_meta.sequence_length,
                    pages_info = [ random.choice( eval_pages ) ],
                    tokenizer = tokenizer
                )
                
                # Compute the losses pre delta.
                print ('Eval pre...')
                pre_delta_losses = []
                for batch in dataset:                
                    input_ids = torch.tensor(batch, dtype=torch.long).to(config.device)
                    labels = input_ids.clone()
                    labels[:, :-1] = input_ids[:, 1:]
                    labels[:, -1] = tokenizer.pad_token_id
                    with torch.no_grad():
                        outputs = master( input_ids=input_ids, labels=labels )
                    pre_delta_losses.append( outputs.loss.item() )                    
                    del input_ids, labels, outputs
                    torch.cuda.empty_cache()

                # Apply delta to master parameters.
                for (name, master_param), (_, delta_param) in zip( master.named_parameters(), miner_delta.named_parameters() ):
                    master_param.data.add_( delta_param.data.to( master.device ) )  # In-place addition
                            
                # Compute the losses post delta.
                print ('Eval post...')
                post_delta_losses = []
                for batch in dataset:                
                    input_ids = torch.tensor(batch, dtype=torch.long).to(config.device)
                    labels = input_ids.clone()
                    labels[:, :-1] = input_ids[:, 1:]
                    labels[:, -1] = tokenizer.pad_token_id
                    with torch.no_grad():
                        outputs = master( input_ids=input_ids, labels=labels )
                    post_delta_losses.append( outputs.loss.item() )                    
                    del input_ids, labels, outputs
                    torch.cuda.empty_cache()
                    
                # Subtract the delta from the model parameters.
                for (name, master_param), (_, delta_param) in zip( master.named_parameters(), miner_delta.named_parameters() ):
                    master_param.data.sub_( delta_param.data.to( master.device ) )  # In-place subtraction.
                
                # Set weights: the difference between the pre and post loss for the models gradients.
                pre_loss = np.mean( pre_delta_losses )
                post_loss = np.mean( post_delta_losses )
                loss_dif = pre_loss - post_loss
                weights[ miner_uid ] = loss_dif
                print ( 'UID', miner_meta.uid, 'loss_dif', loss_dif, 'pre loss', pre_loss, 'post loss', post_loss )
                if config.use_wandb: wandb.log({ "loss_dif": loss_dif, 'pre loss':  pre_loss, 'post loss': post_loss } )
                
                # Check if the master has changed.
                latest_master_meta = get_latest_metadata( master_uid, metagraph, subtensor, CLIENT = CLIENT )
                if master_hash != latest_master_meta.model_hash:
                    print ('A new master has been uploaded, break the loop.')
                    break 

            # # Download the deltas and attain the new master.
            # print ('Sync the new state.')
            # for delta_meta in latest_master_meta.deltas:
            #     delta = download_model( metadata = SimpleNamespace( **delta_meta ), device = 'cpu', CLIENT = CLIENT )
            #     for (name, master_param), (_, delta_param) in zip( master.named_parameters(), delta.named_parameters() ):
            #         master_param.data.add_( delta_param.data.to( master.device ) / len( latest_master_meta.deltas ) ) # Normalized.
                         
        # Handle keyboard interrupts, stops training gracefully.
        except (KeyboardInterrupt, SystemExit):
            handle_exit_signal( None, None, config )
            break
        
        # Handle unknown exceptions, continue training after 5 seconds.
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
            time.sleep(5)
            continue

# Main function.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Miner script')
    parser.add_argument('--name', type=str, default=None, help='Optional name')
    parser.add_argument('--netuid', type=int, default=212, help='Bittensor network uid.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--eval_window', type=int, default=3, help='Number of pages to load')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights and Biases for logging')
    bt.wallet.add_args( parser )
    bt.subtensor.add_args( parser )
    config = bt.config( parser )   
    main( config ) 