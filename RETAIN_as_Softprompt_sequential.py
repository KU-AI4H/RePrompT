import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from llm2vec.models import LlamaBiModel
from huggingface_hub import login, snapshot_download
#https://huggingface.co/alpha-ai/Medical-Guide-COT-llama3.2-1B/tree/main
class RETAIN_LLM_SoftPrompt(nn.Module):
    def __init__(self,
                 llm_model_name="gpt2",
                 patient_emb_dim=256,
                 soft_prompt_len=5,
                 freeze_llm=True):
        super(RETAIN_LLM_SoftPrompt, self).__init__()

        # Load pretrained LLM
        #self.llm = AutoModelForCausalLM.from_pretrained(llm_model_name)
        self.llm = LlamaBiModel.from_pretrained(
            #llm_model_name,
            "knowledgator/Llama-encoder-1.0B",
            device_map="auto",  # <- sharded across all GPUs
            #max_memory=max_mem,
            #torch_dtype=desired_dtype,

            low_cpu_mem_usage=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained('knowledgator/Llama-encoder-1.0B')#llm_model_name
        #self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is None or self.tokenizer.pad_token == '' or self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.llm.config.pad_token_id = self.tokenizer.eos_token_id
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self.llm.resize_token_embeddings(len(self.tokenizer))
                self.llm.config.pad_token_id = self.tokenizer.pad_token_id
        # Soft prompt embeddings (trainable)
        self.soft_prompt_len = soft_prompt_len
        self.soft_prompt = nn.Parameter(
            torch.randn(soft_prompt_len, self.llm.config.hidden_size)
        )

        # Project patient embedding → LLM hidden size
        self.patient_proj = nn.Linear(patient_emb_dim, self.llm.config.hidden_size*soft_prompt_len)
        #self.ln_proj = nn.Linear(self.llm.config.hidden_size, self.llm.config.hidden_size*soft_prompt_len).to(self.llm.device)
        self.ln_proj = nn.ModuleDict()
        for index in range(0, 6):
            self.ln_proj[str(index)] = nn.Linear(self.llm.config.hidden_size, self.llm.config.hidden_size*soft_prompt_len).to(self.llm.device)
        # Classification head
        self.output_layer = nn.Linear(self.llm.config.hidden_size, patient_emb_dim).to(self.llm.device)

        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()

        if freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad = False
    def get_device(self):
        return self.llm.device
    def get_sum(self, layer, my_name):
        grad_sums = {name: p.grad.sum().item()
                     for name, p in layer.named_parameters() if p.grad is not None}
        print(my_name, grad_sums)
    def get_sum_grad(self):
        pass
        #self.get_sum(self.output_layer,'output_layer')
        #self.get_sum(self.ln_proj, 'ln_proj')
        #self.get_sum(self.patient_proj, 'patient_proj')
    def pad_lists_with_mask(self, list_of_lists):
        max_len = max(len(sublist) for sublist in list_of_lists)

        padded = []
        mask = []

        for sublist in list_of_lists:
            pad_len = max_len - len(sublist)
            padded.append(sublist + [''] * pad_len)
            mask.append([1] * len(sublist) + [0] * pad_len)

        return np.array(padded), np.array(mask), max_len

    def forward(self, patient_emb_matrix, kwargs):
        device = self.llm.device

        # Use pre-tokenized tensors if provided; else fallback to current path once (still faster)
        enc_input_ids_list = kwargs.get("enc_input_ids", None)  # list len=T: [B, L_t]
        enc_attn_mask_list = kwargs.get("enc_attn_mask", None)  # list len=T: [B, L_t]
        time_mask_bt = kwargs.get("time_mask_bt", None)  # [B, T] bool
        texts = kwargs.get("llm_output")
        #print(texts)
        if enc_input_ids_list is None:
            # Fallback: one-time tokenization on CPU for all time steps, no NumPy
            # Pad each visit list to same T, and build time_mask_bt
            max_len_t = max(len(v) for v in texts)
            T = min(6, max_len_t)
            B = len(texts)

            # build per-step lists
            step_texts = []
            time_mask_cols = []
            for t in range(T):
                col = []
                mask_col = []
                for b in range(B):
                    if t < len(texts[b]):
                        visit_text = texts[b][t]
                        if t == len(texts[b]) - 1:
                            visit_text = "<last> " + visit_text
                        col.append(visit_text)
                        mask_col.append(1)
                    else:
                        col.append("")  # will become [PAD] after tokenization
                        mask_col.append(0)
                step_texts.append(col)
                time_mask_cols.append(mask_col)

            enc_input_ids_list, enc_attn_mask_list = [], []
            for col in step_texts:
                enc = self.tokenizer(col, return_tensors="pt", padding=True, truncation=True)
                enc_input_ids_list.append(enc["input_ids"])
                enc_attn_mask_list.append(enc["attention_mask"])
            time_mask_bt = torch.tensor(time_mask_cols, dtype=torch.bool).T  # [B, T]

        else:
            T = len(enc_input_ids_list)
            T = min(6, T)
            # assume enc_input_ids_list[t], enc_attn_mask_list[t] are CPU tensors
            # and time_mask_bt is [B,T] bool

        # Move once to device (use non_blocking when DataLoader uses pin_memory=True)
        enc_input_ids_list = [x.to(device, non_blocking=True) for x in enc_input_ids_list[:T]]
        enc_attn_mask_list = [x.to(device, non_blocking=True) for x in enc_attn_mask_list[:T]]
        time_mask_bt = time_mask_bt[:, :T].to(device, non_blocking=True)

        B = patient_emb_matrix.size(0)
        H = self.llm.config.hidden_size
        S = self.soft_prompt_len

        add_lists = []
        # pre-alloc once
        time_series_embed = torch.zeros((B, S, H), device=device, dtype=self.patient_proj.weight.dtype)

        # LLM is frozen → inference context + autocast
        use_inference_ctx = torch.inference_mode if all(
            not p.requires_grad for p in self.llm.parameters()) else torch.no_grad
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):  # or float16
            get_emb = self.llm.get_input_embeddings()

            for t in range(T):
                patient_emb = patient_emb_matrix[:, t, :]  # [B, D]
                input_ids = enc_input_ids_list[t]  # [B, L_t]
                tok_mask = enc_attn_mask_list[t]  # [B, L_t]

                # embed once; stays on device
                input_embeds = get_emb(input_ids)  # [B, L_t, H]

                # project patient prompt
                patient_prompt = self.patient_proj(patient_emb).view(B, S, H)  # [B, S, H]

                # concat prompts + prev time summary + tokens
                inputs_embeds = torch.cat([patient_prompt.to(device), time_series_embed.to(device), input_embeds.to(device)],
                                          dim=1)  # [B, S+S+L_t, H]

                # attention mask (bool saves memory)
                soft_mask = torch.ones((B, S), dtype=torch.bool, device=device)
                soft_timeseries_on = torch.ones((B, S), dtype=torch.bool, device=device)
                soft_timeseries_off = torch.zeros((B, S), dtype=torch.bool, device=device)
                soft_timeseries_mask = soft_timeseries_off if t == 0 else soft_timeseries_on
                full_attn_mask = torch.cat([soft_mask, soft_timeseries_mask, tok_mask.bool()], dim=1)  # [B, 2S+L_t]

                # forward (only need last layer)
                out = self.llm(inputs_embeds=inputs_embeds, attention_mask=full_attn_mask,
                               output_hidden_states=False)
                last_hidden = out.last_hidden_state  # [B, 2S+L_t, H]

                # drop the 2S prompt tokens
                pooled_hidden = last_hidden[:, 2 * S:, :]  # [B, L_t, H]

                # mean pool over valid tokens
                denom = tok_mask.sum(dim=1).clamp_min(1).unsqueeze(1)  # [B,1]
                mean_pooled = (pooled_hidden * tok_mask.unsqueeze(-1)).sum(dim=1) / denom  # [B, H]
                add_lists.append(mean_pooled.to(dtype=self.output_layer.weight.dtype))

                # update time-series prompt for next step
                if t < T - 1:
                    time_series_embed = self.ln_proj[str(t)](mean_pooled).view(B, S, H)

        # pick last valid step per sequence using time_mask_bt
        # convert mask once and compute last index without NumPy
        lengths = time_mask_bt.sum(dim=1)  # [B]
        last_idx = (lengths - 1).clamp_min(0)  # [B]

        result = torch.stack(add_lists, dim=1)  # [B, T, H]
        logits_in = result[torch.arange(B, device=device), last_idx]  # [B, H]
        logits = self.output_layer(logits_in)  # [B, patient_emb_dim]
        return logits
    '''
    def forward(self, patient_emb_matrix, kwargs):



        texts = kwargs["llm_output"]
        #print('pat_embed', patient_emb_matrix.size())

        texts, mask, max_len = self.pad_lists_with_mask(texts)
        add_lists = []
        max_len = min(6,max_len)
        mask = mask[:, :max_len]
        for index in range(0, max_len):
            patient_emb = patient_emb_matrix[:,index,:]
            device = patient_emb.device
            #print('npshape', np.shape(texts[:, index]))
            encodings = self.tokenizer(
                texts[:, index].tolist(), return_tensors="pt", padding=True, truncation=True
            ).to(device)
            input_ids = encodings["input_ids"]  # [B, L]
            tok_mask = encodings["attention_mask"]  # [B, L] (1=real, 0=pad)

            input_embeds = self.llm.get_input_embeddings()(input_ids)  # [B, L, H]
            H = input_embeds.size(-1)

            patient_prompt = self.patient_proj(patient_emb).view(
                patient_emb.size(0), self.soft_prompt_len, H
            )  # [B, S, H], S=self.soft_prompt_len

            # ----- concat: [soft prompts | token embeddings] -----
            if index == 0:
                time_series_embed = torch.zeros((len(texts), self.soft_prompt_len, self.llm.config.hidden_size)).to(self.llm.device)
            #print('index', index)
            #print(patient_prompt.size(), time_series_embed.size(), input_embeds.size())
            inputs_embeds = torch.cat([patient_prompt.to(self.llm.device), time_series_embed, input_embeds], dim=1)  # [B, S+L, H]

            # ----- build attention mask that includes soft prompts -----
            soft_mask = torch.ones(
                (tok_mask.size(0), self.soft_prompt_len),
                dtype=tok_mask.dtype, device=tok_mask.device
            )  # [B, S]
            if index == 0:
                soft_timeseries_mask = torch.zeros(
                    (tok_mask.size(0), self.soft_prompt_len),
                    dtype=tok_mask.dtype, device=tok_mask.device
                )  # [B, S]
            else:
                soft_timeseries_mask = torch.ones(
                    (tok_mask.size(0), self.soft_prompt_len),
                    dtype=tok_mask.dtype, device=tok_mask.device
                )
            full_attn_mask = torch.cat([soft_mask, soft_timeseries_mask, tok_mask], dim=1)  # [B, S+L]

            # ----- forward through LLM with proper attention mask -----
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=full_attn_mask,
                output_hidden_states=True
            )

            # ----- get last hidden states, drop soft prompts -----
            last_hidden = outputs.last_hidden_state#.hidden_states[-1]  # [B, S+L, H]
            pooled_hidden = last_hidden[:, 2*self.soft_prompt_len:, :]  # [B, L, H]

            # ----- mean-pool over ONLY valid tokens from tokenizer mask -----
            #valid_mask = full_attn_mask.to(self.llm.device)   # [B, L] #tok_mask
            valid_mask = tok_mask.to(self.llm.device)
            denom = valid_mask.sum(dim=1, keepdim=True).clamp_min(1).to(self.llm.device)   # avoid /0
            mean_pooled = (pooled_hidden.to(self.llm.device) * valid_mask.unsqueeze(-1)).sum(dim=1).to(self.llm.device) / denom  # [B, H]
            add_lists.append(mean_pooled)
            if index < max_len - 1:
                time_series_embed = self.ln_proj(mean_pooled)
                time_series_embed = time_series_embed.view(-1, self.soft_prompt_len, self.llm.config.hidden_size)
            #print(torch.sum(mean_pooled))
            # ----- head -----
            #logits = self.output_layer(mean_pooled)
        result = torch.stack(add_lists, dim=1)
        last_indices = torch.from_numpy(mask).to(self.llm.device).sum(dim=1) - 1  # (B,)
        B = result.size(0)
        logits = result[torch.arange(B, device=result.device), last_indices]
        logits = self.output_layer(logits)
        return logits
    '''


from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

from pyhealth.datasets import SampleEHRDataset
from pyhealth.models import BaseModel

# VALID_OPERATION_LEVEL = ["visit", "event"]


class RETAINLayer(nn.Module):
    def __init__(
        self,
        feature_size: int,
        dropout: float = 0.5,
    ):
        super(RETAINLayer, self).__init__()
        self.feature_size = feature_size
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        self.alpha_gru = nn.GRU(feature_size, feature_size, batch_first=True)
        self.beta_gru = nn.GRU(feature_size, feature_size, batch_first=True)

        self.alpha_li = nn.Linear(feature_size, 1)
        self.beta_li = nn.Linear(feature_size, feature_size)

    @staticmethod
    def reverse_x(input, lengths):
        """Reverses the input."""
        reversed_input = input.new(input.size())
        for i, length in enumerate(lengths):
            reversed_input[i, :length] = input[i, :length].flip(dims=[0])
        return reversed_input

    def compute_alpha(self, rx, lengths):
        """Computes alpha attention."""
        rx = rnn_utils.pack_padded_sequence(
            rx, lengths, batch_first=True, enforce_sorted=False
        )
        g, _ = self.alpha_gru(rx)
        g, _ = rnn_utils.pad_packed_sequence(g, batch_first=True)
        attn_alpha = torch.softmax(self.alpha_li(g), dim=1)
        return attn_alpha

    def compute_beta(self, rx, lengths):
        """Computes beta attention."""
        rx = rnn_utils.pack_padded_sequence(
            rx, lengths, batch_first=True, enforce_sorted=False
        )
        h, _ = self.beta_gru(rx)
        h, _ = rnn_utils.pad_packed_sequence(h, batch_first=True)
        attn_beta = torch.tanh(self.beta_li(h))
        return attn_beta

    def forward(
        self,
        x: torch.tensor,
        mask: Optional[torch.tensor] = None,
    ) -> Tuple[torch.tensor, torch.tensor]:

        x = self.dropout_layer(x)
        batch_size = x.size(0)
        if mask is None:
            lengths = torch.full(
                size=(batch_size,), fill_value=x.size(1), dtype=torch.int64
            )
        else:
            lengths = torch.sum(mask.int(), dim=-1).cpu()
        rx = self.reverse_x(x, lengths)
        attn_alpha = self.compute_alpha(rx, lengths)
        attn_beta = self.compute_beta(rx, lengths)
        c = attn_alpha * attn_beta * x  # (patient, sequence len, feature_size)
        c = torch.sum(c, dim=1)  # (patient, feature_size)
        return c


class RETAIN(BaseModel):
    def __init__(
        self,
        dataset: SampleEHRDataset,
        feature_keys: List[str],
        label_key: str,
        mode: str,
        pretrained_emb: str = None,
        embedding_dim: int = 128,
        **kwargs,
    ):
        super(RETAIN, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
            pretrained_emb=pretrained_emb,
        )
        self.embedding_dim = embedding_dim

        # validate kwargs for RETAIN layer
        if "feature_size" in kwargs:
            raise ValueError("feature_size is determined by embedding_dim")

        # the key of self.feat_tokenizers only contains the code based inputs
        self.feat_tokenizers = {}
        self.label_tokenizer = self.get_label_tokenizer()
        # the key of self.embeddings only contains the code based inputs
        self.embeddings = nn.ModuleDict()
        # the key of self.linear_layers only contains the float/int based inputs
        self.linear_layers = nn.ModuleDict()

        # add feature RETAIN layers
        for feature_key in self.feature_keys:
            input_info = self.dataset.input_info[feature_key]
            # sanity check
            if input_info["type"] not in [str, float, int]:
                raise ValueError(
                    "RETAIN only supports str code, float and int as input types"
                )
            elif (input_info["type"] == str) and (input_info["dim"] not in [2, 3]):
                raise ValueError(
                    "RETAIN only supports 2-dim or 3-dim str code as input types"
                )
            elif (input_info["type"] in [float, int]) and (
                input_info["dim"] not in [2, 3]
            ):
                raise ValueError(
                    "RETAIN only supports 2-dim or 3-dim float and int as input types"
                )
            # for code based input, we need Type
            # for float/int based input, we need Type, input_dim
            self.add_feature_transform_layer(feature_key, input_info)

        self.retain = nn.ModuleDict()
        for feature_key in feature_keys:
            self.retain[feature_key] = RETAINLayer(feature_size=embedding_dim, **kwargs)

        output_size = self.get_output_size(self.label_tokenizer)

        self.llm_model = RETAIN_LLM_SoftPrompt(llm_model_name='/home/a053h213/PycharmProjects/EHRLLMGraph/raw_llama_1b', patient_emb_dim = embedding_dim* len(self.feature_keys), soft_prompt_len=8, freeze_llm=True)
        self.fc = nn.Linear(len(self.feature_keys) * self.embedding_dim, output_size).to(self.llm_model.get_device())
    def index_retain(self, index, **kwargs):
        patient_emb = []
        for feature_key in self.feature_keys:
            input_info = self.dataset.input_info[feature_key]
            dim_, type_ = input_info["dim"], input_info["type"]

            # for case 1: [code1, code2, code3, ...]
            if (dim_ == 2) and (type_ == str):
                x = self.feat_tokenizers[feature_key].batch_encode_2d(
                    kwargs[feature_key]
                )
                # (patient, event)
                x = torch.tensor(x, dtype=torch.long, device=self.device)
                # (patient, event, embedding_dim)
                x = self.embeddings[feature_key](x)
                # (patient, event)
                mask = torch.sum(x, dim=2) != 0

            # for case 2: [[code1, code2], [code3, ...], ...]
            elif (dim_ == 3) and (type_ == str):
                x = self.feat_tokenizers[feature_key].batch_encode_3d(
                    kwargs[feature_key]
                )
                # (patient, visit, event)
                x = torch.tensor(x, dtype=torch.long, device=self.device)
                # (patient, visit, event, embedding_dim)
                x = self.embeddings[feature_key](x)
                x = x[:, :index+1, :]
                # (patient, visit, embedding_dim)
                x = torch.sum(x, dim=2)
                # (patient, visit)
                mask = torch.sum(x, dim=2) != 0

            # for case 3: [[1.5, 2.0, 0.0], ...]
            elif (dim_ == 2) and (type_ in [float, int]):
                x, mask = self.padding2d(kwargs[feature_key])
                # (patient, event, values)
                x = torch.tensor(x, dtype=torch.float, device=self.device)
                # (patient, event, embedding_dim)
                x = self.linear_layers[feature_key](x)
                # (patient, event)
                mask = mask.bool().to(self.device)

            # for case 4: [[[1.5, 2.0, 0.0], [1.8, 2.4, 6.0]], ...]
            elif (dim_ == 3) and (type_ in [float, int]):
                x, mask = self.padding3d(kwargs[feature_key])
                # (patient, visit, event, values)
                x = torch.tensor(x, dtype=torch.float, device=self.device)
                # (patient, visit, embedding_dim)
                x = torch.sum(x, dim=2)
                x = self.linear_layers[feature_key](x)
                # (patient, event)
                mask = mask[:, :, 0]
                mask = mask.bool().to(self.device)

            else:
                raise NotImplementedError

            # transform x to (patient, event, embedding_dim)
            if self.pretrained_emb != None:
                x = self.linear_layers[feature_key](x)

            x = self.retain[feature_key](x, mask)
            patient_emb.append(x)

        patient_emb = torch.cat(patient_emb, dim=1)
        return patient_emb
    def get_sum_grad(self):
        self.llm_model.get_sum_grad()
    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        x = self.feat_tokenizers['conditions'].batch_encode_3d(
            kwargs['conditions']
        )
        x = torch.tensor(x, dtype=torch.long, device=self.device)
        x = self.embeddings['conditions'](x)
        max = x.size()[1]
        pat_emb_sequence = []
        for index in range(0, max):
            patient_emb = self.index_retain(index, **kwargs)
            pat_emb_sequence.append(patient_emb)
        patient_emb = torch.stack(pat_emb_sequence, dim=1)
        logits = self.llm_model(patient_emb, kwargs)
        logits = self.fc(logits)
        y_true = self.prepare_labels(kwargs[self.label_key], self.label_tokenizer).to(self.llm_model.get_device())
        loss = self.get_loss_function()(logits.to('cuda:0'), y_true.to('cuda:0'))
        y_prob = self.prepare_y_prob(logits).to(self.llm_model.get_device())
        results = {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logits,
        }
        if kwargs.get("embed", False):
            results["embed"] = patient_emb
        return results
