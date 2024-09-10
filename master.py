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
import sys
import copy
import math
import time
import boto3
import torch
import wandb
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

# Main function.
def main( config ):
    print ( config )
    
    # Init Bittensor objects.
    wallet = bt.wallet( config = config )
    subtensor = bt.subtensor( config = config )
    metagraph = subtensor.metagraph( netuid = config.netuid )
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        print(f'\tWallet {wallet} is not registered on subnet: {metagraph.netuid}'); sys.exit()
    my_uid = metagraph.hotkeys.index( wallet.hotkey.ss58_address )
    print ( f'Wallet: {wallet}\nSubtensor: {subtensor}\nMetagraph: {metagraph}\nUID: {my_uid}' )
    
    # Assert the chain commitment.
    try:
        if config.bucket != subtensor.get_commitment(config.netuid, my_uid):
            raise ValueError(f'Chain commitment does not match: {config.bucket}')
    except Exception:
        subtensor.commit( wallet, config.netuid, config.bucket)
    print('Bucket:', config.bucket )
    
    # Build the tokenizer.
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained( config.tokenizer_name, verbose=False, clean_up_tokenization_spaces=True )
    tokenizer.pad_token = tokenizer.eos_token        
    print ('Tokenizer:', config.tokenizer_name)

    # Init model based on type.
    master = None
    my_current_meta = get_latest_metadata( my_uid, metagraph, subtensor, CLIENT = CLIENT )
    if my_current_meta != None and not config.restart:
        # Download the model state from remote.
        print ('Downloading master state from remote.')
        master = download_model( metadata = my_current_meta, device = config.device, CLIENT = CLIENT )
        
    # If the master does not exist, create it.
    if master == None or config.restart:
        # Create a new model
        print ('Restarting model from scratch.')
        if config.model_type == 'gpt2':
            master = GPT2LMHeadModel( config = GPT2Config(
                output_hidden_states = False, 
                n_positions = config.sequence_length
            ))
        elif config.model_type == 'llama':
            master = LlamaForCausalLM( config = LlamaConfig(
                vocab_size = tokenizer.vocab_size,     
                hidden_size = 2040,   
                num_hidden_layers = 12,  
                num_attention_heads = 12,
                intermediate_size = 6144
            ))
            
    # Upload the master stake.
    print (f'Model: {config.model_type}')
    history = [] # Previous model uploads.
    history.append(upload_model(
        wallet = wallet, 
        model = master, 
        block = int(time.time()),
        extras = { 'sequence_length': config.sequence_length, 'tokenizer_name': config.tokenizer_name }, 
        bucket = config.bucket,
        CLIENT = CLIENT
    ))

    # Init weights and biases
    if config.use_wandb:
        name = f'Master-{wallet.hotkey.ss58_address[:5]}' 
        run = wandb.init(project='bistro', resume = 'allow', name = name, config = config )
        
    # Remember delta for later removal.
    steps = 0
    n_accumulated = 0
    while True:
        try:
        
            # Resync chain state.
            master_hash = hash_model( master )
            subtensor = bt.subtensor( config = config )
            metagraph = subtensor.metagraph( netuid = config.netuid )
                                    
            # Pull metadeta from miners and checking for enough to apply.
            pending = []
            for uid in metagraph.uids:
                if uid == my_uid: continue
                delta_meta = get_latest_metadata( uid, metagraph, subtensor, CLIENT = CLIENT )
                if delta_meta == None or delta_meta.master_hash != master_hash:
                    continue
                pending.append( delta_meta )
            if len( pending ) < config.n_accumulations:
                print ('\tWaiting for deltas ...')
                continue
            
            # Got deltas, starting step.
            steps += 1 
            if config.use_wandb: wandb.log({ "step": steps, "block": subtensor.block } )
            print('-' * 80)
            print ( f'Step: {steps}, Hash: {master_hash}, Block: {subtensor.block}' )

            # Download and apply each of the deltas to the master
            print (f'\tApplying {len(pending)} deltas ...')
            applied = []
            for delta_meta in pending:
                
                # Download the delta.
                delta = download_model( metadata = delta_meta, device = 'cpu', CLIENT = CLIENT )
                if delta != None:
                    
                    # Apply the delta to the master
                    for (name, master_param), (_, delta_param) in zip( master.named_parameters(), delta.named_parameters() ):
                        # Sanatize the delta NaN vales.
                        delta_update = delta_param.data.to( master.device )
                        if torch.isnan(delta_update).any():
                            delta_update[ torch.isnan(delta_update) ] = 0  # Set NaNs to 0
                        master_param.data.add_( delta_update / len( pending ) ) 
                    
                    # Check if the model has NaNs after applying the delta
                    has_nans = any(torch.isnan(param).any() for param in master.parameters())
                    if has_nans:
                        print("Warning: The model contains NaNs after applying the delta.")
                        # Remove NaNs from the master parameters
                        for param in master.parameters():
                            param.data[torch.isnan(param.data)] = 0
                    applied.append( delta_meta.__dict__ )
                    
                    # Increment counter.
                    n_accumulated += 1
                    if config.use_wandb: wandb.log({ "n_accumulated": n_accumulated } )
            
            # Upload the new master.
            # TODO: We should be uploading the delta to save bandwidth rather than have the miners query the other S3 buckets.
            history.append( upload_model(
                wallet = wallet, 
                model = master, 
                block = int(time.time()),
                extras = { 'sequence_length': config.sequence_length, 'tokenizer_name': config.tokenizer_name, 'deltas': applied }, 
                bucket = config.bucket,
                CLIENT = CLIENT
            ))
            if len(history) > config.history_size:
                old_model = history.pop(0)
                print (f'Deleting old model: {old_model.filename}, still holding: {len(history)} files.')
                CLIENT.delete_object( Bucket = config.bucket, Key = old_model.filename )
                CLIENT.delete_object( Bucket = config.bucket, Key = old_model.metadata_filename )

                                        
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
            print(f"Error: {e}")
            traceback.print_exc()
            time.sleep(5)
            continue
                    

# Main function.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Miner script')
    parser.add_argument('--name', type=str, default=None, help='Optional name')
    parser.add_argument('--netuid', type=int, default=212, help='Bittensor network uid.')
    parser.add_argument('--bucket', type=str, default='decis', help='S3 bucket name')
    parser.add_argument('--history_size', type=int, default=2, help='Number of previous models to maintain.')
    parser.add_argument('--sequence_length', type=int, default=2048, help='Sequence Length.')
    parser.add_argument('--tokenizer_name', type=str, default='gpt2', help='Tokenizer name.')
    parser.add_argument('--n_accumulations', type=int, default=1, help='Number of steps before we upload a new model state.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights and Biases for logging')
    parser.add_argument('--model_type', type=str, choices=['gpt2', 'llama'], default='gpt2', help='Model type to use: gpt2 or llama')
    parser.add_argument('--restart', action='store_true', default=False, help='Restart with a new training state.')
    bt.wallet.add_args( parser )
    bt.subtensor.add_args( parser )
    config = bt.config( parser )   
    main( config ) 