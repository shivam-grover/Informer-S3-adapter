import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding
from models.S3Layer import S3Layer

class AdapterLayer(nn.Module):
    def __init__(self, input_dim, adapter_dim):
        super(AdapterLayer, self).__init__()
        self.down_proj = nn.Linear(input_dim, adapter_dim)
        self.activation = nn.ReLU()
        self.up_proj = nn.Linear(adapter_dim, input_dim)

    def forward(self, x):
        return self.up_proj(self.activation(self.down_proj(x))) + x

class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0'),
                num_segments=4, num_S3_stacks=1, consecutive_segment_num_ratio=0.5, shuffle_vector_dim=2, S3_kernel_size=3, enable_S3=1,
                adapter_dim=64):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        print(adapter_dim)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                    adapter_layer=AdapterLayer(d_model, adapter_dim)
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                    adapter_layer=AdapterLayer(d_model, adapter_dim)
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
        self.enable_S3 = int(enable_S3)
        self.num_segments=num_segments
        self.sample_num_to_truncate = 0
        self.num_S3_stacks = num_S3_stacks
        self.shuffle_vector_dim = shuffle_vector_dim
        
        # if self.enable_S3==1:
        #     self.S3Layer_enc = S3Layer_channelwise(num_channels=self.in_dim, num_segments=self.num_segments, shuffle_vector_dim=self.shuffle_vector_dim, kernel_size=S3_kernel_size)
        
        # if self.enable_S3==1:
        #     self.S3Layer_dec = S3Layer_channelwise(num_channels=self.in_dim, num_segments=self.num_segments, shuffle_vector_dim=self.shuffle_vector_dim, kernel_size=S3_kernel_size)

        if self.enable_S3==1:
            self.S3Layer_enc = S3Layer(num_segments=self.num_segments, shuffle_vector_dim=self.shuffle_vector_dim)
        
        if self.enable_S3==1:
            self.S3Layer_dec = S3Layer(num_segments=self.num_segments, shuffle_vector_dim=self.shuffle_vector_dim)
        self.freeze_pretrained_layers()

    def freeze_pretrained_layers(self):
        for param in self.enc_embedding.parameters():
            param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.dec_embedding.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False
        for param in self.projection.parameters():
            param.requires_grad = False
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        if self.enable_S3==1:
            ###########################################################################
            # Inject S3 layer
            x_enc_copy = x_enc.clone().to(x_enc.device)

            sample_num_to_truncate = 0
            if(x_enc_copy.shape[1] % self.S3Layer_enc.num_segments!=0):
                sample_num_to_truncate = x_enc_copy.shape[1] % self.S3Layer_enc.num_segments
            if(sample_num_to_truncate!=0):
                x_enc_copy = x_enc_copy[:, sample_num_to_truncate:, :]
            
            x_enc_copy = self.S3Layer_enc(x_enc_copy)
            x_enc_copy = torch.cat([x_enc[:, 0:(x_enc.shape[1] - x_enc_copy.shape[1]), :], x_enc_copy], dim=1)
            x_enc = x_enc_copy
            ###########################################################################
            x_dec_copy = x_dec.clone().to(x_dec.device)
            # print(x_dec_copy.shape, x_dec.shape)

            sample_num_to_truncate = 0
            if(x_dec_copy.shape[1] % self.S3Layer_dec.num_segments!=0):
                sample_num_to_truncate = x_dec_copy.shape[1] % self.S3Layer_dec.num_segments
            if(sample_num_to_truncate!=0):
                x_dec_copy = x_dec_copy[:, sample_num_to_truncate:, :]
            # print(x_dec_copy.shape)
            
            x_dec_copy = self.S3Layer_dec(x_dec_copy)
            x_dec_copy = torch.cat([x_dec[:, 0:(x_dec.shape[1] - x_dec_copy.shape[1]), :], x_dec_copy], dim=1)
            x_dec = x_dec_copy
            ###########################################################################
            x_mark_enc_copy = x_mark_enc.clone().to(x_mark_enc.device)

            sample_num_to_truncate = 0
            if(x_mark_enc_copy.shape[1] % self.S3Layer_enc.num_segments!=0):
                sample_num_to_truncate = x_mark_enc_copy.shape[1] % self.S3Layer_enc.num_segments
            if(sample_num_to_truncate!=0):
                x_mark_enc_copy = x_mark_enc_copy[:, sample_num_to_truncate:, :]
            
            x_mark_enc_copy = self.S3Layer_enc(x_mark_enc_copy)
            x_mark_enc_copy = torch.cat([x_mark_enc[:, 0:(x_mark_enc.shape[1] - x_mark_enc_copy.shape[1]), :], x_mark_enc_copy], dim=1)
            x_mark_enc = x_mark_enc_copy
            ###########################################################################
            x_mark_dec_copy = x_mark_dec.clone().to(x_mark_dec.device)

            sample_num_to_truncate = 0
            if(x_mark_dec_copy.shape[1] % self.S3Layer_dec.num_segments!=0):
                sample_num_to_truncate = x_mark_dec_copy.shape[1] % self.S3Layer_dec.num_segments
            if(sample_num_to_truncate!=0):
                x_mark_dec_copy = x_mark_dec_copy[:, sample_num_to_truncate:, :]
            
            x_mark_dec_copy = self.S3Layer_dec(x_mark_dec_copy)
            x_mark_dec_copy = torch.cat([x_mark_dec[:, 0:(x_mark_dec.shape[1] - x_mark_dec_copy.shape[1]), :], x_mark_dec_copy], dim=1)
            x_mark_dec = x_mark_dec_copy
            ###########################################################################
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:] # [B, L, D]


class InformerStack(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=[3,2,1], d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        super(InformerStack, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder

        inp_lens = list(range(len(e_layers))) # [0,1,2,...] you can customize here
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                    d_model, n_heads, mix=False),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for l in range(el-1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in e_layers]
        self.encoder = EncoderStack(encoders, inp_lens)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:] # [B, L, D]
