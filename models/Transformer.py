"""
Transformer model.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import numpy as np

import torch
from torch import nn
import random

####### Do not modify these imports.

class TransformerTranslator(nn.Module):
    """
    A single-layer Transformer which encodes a sequence of text and 
    performs binary classification.

    The model has a vocab size of V, works on
    sequences of length T, has an hidden dimension of H, uses word vectors
    also of dimension H, and operates on minibatches of size N.
    """
    def __init__(self, input_size, output_size, device, hidden_dim=128, num_heads=2,
                  dim_feedforward=2048, dim_k=96, dim_v=96, dim_q=96, max_length=43):
        """
        :param input_size: the size of the input, which equals to the number of words in source language vocabulary
        :param output_size: the size of the output, which equals to the number of words in target language vocabulary
        :param hidden_dim: the dimensionality of the output embeddings that go into the final layer
        :param num_heads: the number of Transformer heads to use
        :param dim_feedforward: the dimension of the feedforward network model
        :param dim_k: the dimensionality of the key vectors
        :param dim_q: the dimensionality of the query vectors
        :param dim_v: the dimensionality of the value vectors
        """
        super(TransformerTranslator, self).__init__()
        assert hidden_dim % num_heads == 0
        
        self.num_heads = num_heads
        self.word_embedding_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.max_length = max_length
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dim_q = dim_q
        
        seed_torch(0)
        
        ##############################################################################
        # TODO:
        # Deliverable 1: Initialize what you need for the embedding lookup.          #
        # You will need to use the max_length parameter above.                       #
        # Don’t worry about sine/cosine encodings- use positional encodings.         #
        ##############################################################################
        self.embeddingL = nn.Embedding(input_size, hidden_dim).to(device)
        self.posembeddingL = nn.Embedding(max_length, hidden_dim).to(device)
        
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        
        
        ##############################################################################
        # Deliverable 2: Initializations for multi-head self-attention.              #
        # You don't need to do anything here. Do not modify this code.               #
        ##############################################################################
        
        # Head #1
        self.k1 = nn.Linear(self.hidden_dim, self.dim_k).to(device)
        self.v1 = nn.Linear(self.hidden_dim, self.dim_v).to(device)
        self.q1 = nn.Linear(self.hidden_dim, self.dim_q).to(device)
        
        self.k2 = nn.Linear(self.hidden_dim, self.dim_k).to(device)
        self.v2 = nn.Linear(self.hidden_dim, self.dim_v).to(device)
        self.q2 = nn.Linear(self.hidden_dim, self.dim_q).to(device)
        
        self.softmax = nn.Softmax(dim=2)
        self.attention_head_projection = nn.Linear(self.dim_v * self.num_heads, self.hidden_dim).to(device)
        self.norm_mh = nn.LayerNorm(self.hidden_dim).to(device)
        

        
        ##############################################################################
        # TODO:
        # Deliverable 3: Initialize what you need for the feed-forward layer.        # 
        # Don't forget the layer normalization.                                      #
        ##############################################################################
        self.feedforward1 = nn.Linear(hidden_dim, dim_feedforward).to(device)
        self.feedforward2 = nn.Linear(dim_feedforward, hidden_dim).to(device)
        self.norm_ff = nn.LayerNorm(hidden_dim).to(device)
        self.relu = nn.ReLU()
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        
        ##############################################################################
        # TODO:
        # Deliverable 4: Initialize what you need for the final layer (1-2 lines).   #
        ##############################################################################
        self.final = nn.Linear(hidden_dim, output_size).to(device)
        
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        
    def forward(self, inputs):
        """
        This function computes the full Transformer forward pass.
        Put together all of the layers you've developed in the correct order.

        :param inputs: a PyTorch tensor of shape (N,T). These are integer lookups.

        :returns: the model outputs. Should be scores of shape (N,T,output_size).
        """

        #############################################################################
        # TODO:
        # Deliverable 5: Implement the full Transformer stack for the forward pass. #
        # You will need to use all of the methods you have previously defined above.#
        # You should only be calling TransformerTranslator class methods here.      #
        #############################################################################
        inputs.to(self.device)
        #print('inputs.device:', inputs.device)
        embedded = self.embed(inputs)
        attention_output = self.multi_head_attention(embedded)
        feedforward_output = self.feedforward_layer(attention_output)
        outputs = self.final_layer(feedforward_output)
        
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
    
    
    def embed(self, inputs):
        """
        :param inputs: intTensor of shape (N,T)
        :returns embeddings: floatTensor of shape (N,T,H)
        """
        #############################################################################
        # TODO:
        # Deliverable 1: Implement the embedding lookup.                            #
        # Note: word_to_ix has keys from 0 to self.vocab_size - 1                   #
        # This will take a few lines.                                               #
        #############################################################################
        inputs.to(self.device)
        #print('self.device:',self.device)
        #print('inputs.device:', inputs.device)
        batch_size = inputs.shape[0]
        seq_length = inputs.shape[1]
        
        word_embeddings = self.embeddingL(inputs).to(self.device)
        positions = torch.arange(0, seq_length).expand(batch_size, seq_length).to(self.device)
        pos_embeddings = self.posembeddingL(positions).to(self.device)
        
        embeddings = word_embeddings + pos_embeddings
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return embeddings
        
    def multi_head_attention(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        
        Traditionally we'd include a padding mask here, so that pads are ignored.
        This is a simplified implementation.
        """
        
        
        #############################################################################
        # TODO:
        # Deliverable 2: Implement multi-head self-attention followed by add + norm.#
        # Use the provided 'Deliverable 2' layers initialized in the constructor.   #
        #############################################################################
        k1 = self.k1(inputs)
        v1 = self.v1(inputs)
        q1 = self.q1(inputs)
        
        k2 = self.k2(inputs)
        v2 = self.v2(inputs)
        q2 = self.q2(inputs)
        
        scores1 = torch.bmm(q1, k1.transpose(1, 2)) / np.sqrt(self.dim_k)
        attention1 = self.softmax(scores1)
        head1 = torch.bmm(attention1, v1)
        
        scores2 = torch.bmm(q2, k2.transpose(1, 2)) / np.sqrt(self.dim_k)
        attention2 = self.softmax(scores2)
        head2 = torch.bmm(attention2, v2)

        concat_heads = torch.cat((head1, head2), dim=2)
        multi_head_output = self.attention_head_projection(concat_heads)
        attention3 = self.softmax(scores2)
        outputs = self.norm_mh(multi_head_output + inputs)
        
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
    
    
    def feedforward_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        """
        
        #############################################################################
        # TODO:
        # Deliverable 3: Implement the feedforward layer followed by add + norm.    #
        # Use a ReLU activation and apply the linear layers in the order you        #
        # initialized them.                                                         #
        # This should not take more than 3-5 lines of code.                         #
        #############################################################################
        ff_output = self.feedforward1(inputs)
        ff_output = self.relu(ff_output)
        ff_output = self.feedforward2(ff_output)
        outputs = self.norm_ff(ff_output + inputs)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
        
    
    def final_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,V)
        """
        
        #############################################################################
        # TODO:
        # Deliverable 4: Implement the final layer for the Transformer Translator.  #
        # This should only take about 1 line of code. Softmax is not needed here    #
        # as it is integrated as part of cross entropy loss function.               #
        #############################################################################
        outputs = self.final(inputs)     
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
        

class FullTransformerTranslator(nn.Module):

    def __init__(self, input_size, output_size, device, hidden_dim=128, num_heads=2,
                 dim_feedforward=2048, num_layers_enc=2, num_layers_dec=2, dropout=0.2, max_length=43, ignore_index=1):
        super(FullTransformerTranslator, self).__init__()

        self.num_heads = num_heads
        self.word_embedding_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.max_length = max_length
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.pad_idx=ignore_index

        seed_torch(0)

        ##############################################################################
        # TODO:
        # Deliverable 1: Initialize what you need for the Transformer Layer          #
        # You should use nn.Transformer                                              #
        ##############################################################################
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers_enc,
            num_decoder_layers=num_layers_dec,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        ##############################################################################
        # TODO:
        # Deliverable 2: Initialize what you need for the embedding lookup.          #
        # You will need to use the max_length parameter above.                       #
        # Initialize embeddings in order shown below.                                #
        # Don’t worry about sine/cosine encodings- use positional encodings.         #
        ##############################################################################
        # Do not change the order for these variables
        self.srcembeddingL = nn.Embedding(input_size, hidden_dim)
        self.tgtembeddingL = nn.Embedding(output_size, hidden_dim)
        self.srcposembeddingL = nn.Embedding(max_length, hidden_dim)
        self.tgtposembeddingL = nn.Embedding(max_length, hidden_dim)
        ##############################################################################
        # TODO:
        # Deliverable 3: Initialize what you need for the final layer.               #
        ##############################################################################
        self.final = nn.Linear(hidden_dim, output_size)
    
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

    def forward(self, src, tgt):
        """
         This function computes the full Transformer forward pass used during training.
         Put together all of the layers you've developed in the correct order.

         :param src: a PyTorch tensor of shape (N,T) these are tokenized input sentences
                tgt: a PyTorch tensor of shape (N,T) these are tokenized translations
         :returns: the model outputs. Should be scores of shape (N,T,output_size).
         """
        #############################################################################
        # TODO:
        # Deliverable 4: Implement the full Transformer stack for the forward pass. #
        #############################################################################
        
        # shift tgt to right, add one <sos> to the beginning and shift the other tokens to right
        tgt = self.add_start_token(tgt)
        


        # embed src and tgt for processing by transformer
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(self.device)
        tgt_padding_mask = (tgt == self.pad_idx)
        src_padding_mask = (src == self.pad_idx)
        
        # create target mask and target key padding mask for decoder - Both have boolean values
        src_positions = torch.arange(0, src.size(1)).expand(src.size(0), src.size(1)).to(self.device)
        tgt_positions = torch.arange(0, tgt.size(1)).expand(tgt.size(0), tgt.size(1)).to(self.device)
        
        # invoke transformer to generate output
        src_embedded = self.srcembeddingL(src) + self.srcposembeddingL(src_positions)
        tgt_embedded = self.tgtembeddingL(tgt) + self.tgtposembeddingL(tgt_positions)
        
        # pass through final layer to generate outputs
        transformer_out = self.transformer(
            src_embedded,
            tgt_embedded,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        
        # final projection
        outputs = self.final(transformer_out)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs

    def generate_translation(self, src):
        """
         This function generates the output of the transformer taking src as its input
         it is assumed that the model is trained. The output would be the translation
         of the input

         :param src: a PyTorch tensor of shape (N,T)

         :returns: the model outputs. Should be scores of shape (N,T,output_size).
         """
        #############################################################################
        # TODO:
        # Deliverable 5: You will be calling the transformer forward function to    #
        # generate the translation for the input.                                   #
        #############################################################################
        batch_size, seq_len = src.shape

        # Initialize outputs and tgt
        outputs = torch.zeros((batch_size, seq_len, self.output_size), device=self.device)
        tgt = torch.full((batch_size, seq_len), fill_value=self.pad_idx, dtype=torch.long, device=self.device)
        
        # Start decoding with <sos> token
        tgt[:, 0] = 2  # Assuming 2 is the <sos> token index

        for t in range(1, seq_len):
            # Generate predictions for the current target sequence
            output = self.forward(src, tgt[:, :t])

            # Take the highest probability prediction for each batch element at the last time step
            predicted_tokens = output[:, -1, :].argmax(dim=-1)

            # Update tgt with the predicted tokens at the current time step
            tgt[:, t] = predicted_tokens

            # Accumulate the predictions in outputs tensor
            outputs[:, t, :] = output[:, -1, :]

            # Stop if all sequences predict <eos> (assuming <eos> token is index 3)
            if (predicted_tokens == 3).all():  # Assuming 3 is the <eos> token index
                break
        

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs

    def add_start_token(self, batch_sequences, start_token=2):
        """
            add start_token to the beginning of batch_sequence and shift other tokens to the right
            if batch_sequences starts with two consequtive <sos> tokens, return the original batch_sequence

            example1:
            batch_sequence = [[<sos>, 5,6,7]]
            returns:
                [[<sos>,<sos>, 5,6]]

            example2:
            batch_sequence = [[<sos>, <sos>, 5,6,7]]
            returns:
                [[<sos>, <sos>, 5,6,7]]
        """
        def has_consecutive_start_tokens(tensor, start_token):
            """
                return True if the tensor has two consecutive start tokens
            """
            consecutive_start_tokens = torch.tensor([start_token, start_token], dtype=tensor.dtype,
                                                    device=tensor.device)

            # Check if the first two tokens in each sequence are equal to consecutive start tokens
            is_consecutive_start_tokens = torch.all(tensor[:, :2] == consecutive_start_tokens, dim=1)

            # Return True if all sequences have two consecutive start tokens at the beginning
            return torch.all(is_consecutive_start_tokens).item()

        if has_consecutive_start_tokens(batch_sequences, start_token):
            return batch_sequences

        # Clone the input tensor to avoid modifying the original data
        modified_sequences = batch_sequences.clone()

        # Create a tensor with the start token and reshape it to match the shape of the input tensor
        start_token_tensor = torch.tensor(start_token, dtype=modified_sequences.dtype, device=modified_sequences.device)
        start_token_tensor = start_token_tensor.view(1, -1)

        # Shift the words to the right
        modified_sequences[:, 1:] = batch_sequences[:, :-1]

        # Add the start token to the first word in each sequence
        modified_sequences[:, 0] = start_token_tensor

        return modified_sequences

def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True