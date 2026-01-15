"""

"""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Literal


import torch
import torch.nn.functional as F
from tqdm import tqdm

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM
from lm_eval.models.utils_hf import stop_sequences_criteria
from lm_eval.models.utils import (
    Collator,
    _add_special_kwargs,
    configure_pad_token,
    handle_stop_sequences,
    has_bos_prefix,
    postprocess_generated_text,
)
from lm_eval.models.utils_hf import (
    clear_torch_cache,
    get_dtype,
    pad_and_concat,
    stop_sequences_criteria,
)

if TYPE_CHECKING:
    import transformers
    from transformers.quantizers.auto import AutoQuantizationConfig
    from lm_eval.api.instance import Instance

eval_logger = logging.getLogger(__name__)


@register_model("hf-ouro")
class OuroLM(HFLM):
    """
    Model adapter for Ouro models.
    """

    AUTO_MODEL_CLASS = None  # Set dynamically in __init__

    def __init__(self, **kwargs):
        # Import here to avoid import errors if transformers version doesn't support Mistral3
        try:
            from transformers import OuroForCausalLM

            self.AUTO_MODEL_CLASS = OuroForCausalLM
        except ImportError:
            try:
                from ouro.modeling_ouro import OuroForCausalLM
                self.AUTO_MODEL_CLASS = OuroForCausalLM
            except ImportError :
                raise ImportError(
                    "OuroForConditionalGeneration not found in ouro.models. "
                    "Please install ouro >= 0.1.0 or from main: "
                    "pip install git+https://github.com/huggingface/ouro"
                ) from None
        self.ouro_config =  {
            "exit_threshold": kwargs.pop("early_exit_threshold", 1.0),
            "exit_step": kwargs.pop("early_exit_step", None),
            "no_need_keys": kwargs.pop("no_need_keys", ['hidden_states_list', 'gate_list', 'add_noise_list']),#和modeling_ouro中的参数名称一致
        }  
        self.ouro_exit_step_output_dir = kwargs.pop("ouro_exit_step_output_dir", None)
        self.model_name = kwargs.get("model_name", "")
        if self.ouro_exit_step_output_dir is None:
            import os
            self.ouro_exit_step_output_dir = os.environ.get("OUTPUT_EXIT_STEPS_DIR", None)
        super().__init__(**kwargs)
    def _create_model(
        self,
        pretrained: str,
        revision: str | None = "main",
        dtype: str | torch.dtype | None = "auto",
        trust_remote_code: bool | None = False,
        # arguments used for splitting a model across GPUs naively.
        # only used if `parallelize=True`.
        # (accelerate naive PP (device_map) options)
        parallelize: bool | None = False,
        gpus: int | None = None,
        max_memory_per_gpu: int | str | None = None,
        max_cpu_memory: int | str | None = None,
        offload_folder: str | None = "./offload",
        # PEFT, delta weights and quantization options
        peft: str | None = None,
        delta: str | None = None,
        autogptq: bool | str | None = False,
        gptqmodel: bool | None = False,
        quantization_config: AutoQuantizationConfig | None = None,
        gguf_file: str | None = None,
        model_name: str | None = None, #THREEGOLDCHANGE:add model_name
        **kwargs,
    ) -> None:
        """Initializes an HF or HF-compatible PreTrainedModel from scratch
        inside HFLM, using the kwargs passed into self.__init__().

        Also handles functionality such as AutoGPTQ usage and PEFT wrapping.

        For future similar extensions to AutoGPTQ that are not core to HF's ecosystem,
        (such as PyTorch models that are nearly, but not quite, fully mirroring
        HF's public interface relied on in this HFLM class)
        please consider subclassing HFLM and overriding this and other methods as needed.
        """
        from ouro_cache_fix import UniversalTransformerCache 
        cache = UniversalTransformerCache() 
        super()._create_model(pretrained, revision, dtype, trust_remote_code, parallelize, gpus, max_memory_per_gpu, max_cpu_memory, offload_folder, peft, delta, autogptq, gptqmodel, gguf_file,  **kwargs)
    def _create_tokenizer(
        self,
        pretrained: str | transformers.PreTrainedModel,
        tokenizer: str
        | transformers.PreTrainedTokenizer
        | transformers.PreTrainedTokenizerFast
        | None,
        revision: str | None = "main",
        trust_remote_code: bool | None = False,
        use_fast_tokenizer: bool | None = True,
        gguf_file: str | None = None,
        add_bos_token: bool | None = None,
        subfolder: str | None = "",
    ) -> None:
        """Initializes a tokenizer for the model."""
        super()._create_tokenizer(pretrained, tokenizer, revision, trust_remote_code, use_fast_tokenizer, gguf_file, add_bos_token, subfolder)
        self.tokenizer.pad_token = '<|endoftext|>'
        self.tokenizer.eos_token = "<|im_end|>"
        
    def _get_backend(
        self,
        config: transformers.PretrainedConfig | transformers.AutoConfig,
        backend: Literal["default", "causal", "seq2seq"] = "default",
        trust_remote_code: bool | None = False,
    ) -> None:
        """
        Override to force causal backend for Ouro models.

        Ouro models are decoder-only despite using a conditional generation class.
        """ 
        # Always use causal backend for Ouro
        self.backend = "causal"
        eval_logger.info("Using backend 'causal' for Ouro model")
    
    def _get_config(
        self,
        pretrained: str,
        *,
        revision: str = "main",
        trust_remote_code: bool = False,
        gguf_file: str | None = None,
        subfolder: str = "",
    ) -> None:
        """Return the model config for HuggingFace models."""
        
        try:
            from ouro.configuration_ouro import OuroConfig
            self._config = OuroConfig.from_pretrained(
                pretrained,
                revision=revision,
                trust_remote_code=trust_remote_code,
                gguf_file=gguf_file,
                subfolder=subfolder,
            )
        except ImportError:
            raise ImportError(
                "OuroConfig not found in ouro.configuration_ouro. "
                "Please install ouro >= 0.1.0 or from main: "
                "pip install git+https://github.com/huggingface/ouro"
            ) from None
    @property
    def tokenizer_name(self) -> str:
        #THREEGOLDCHANGE:
        import os
        return "_".join(self.tokenizer.name_or_path.split("/")[-2:])
    
    def _model_call(
        self,
        inps: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Override to handle Ouro model output format.

        OuroForCausalLM returns logits in the same format as
        causal LMs, so we call the model directly but bypass the base class
        assertion that checks for AutoModelForCausalLM.
        """
        # 2. 定义专门提取 exit_steps 的钩子函数
        exit_steps_list = []
        def exit_step_hook(module, input, output):
            if hasattr(output, "exit_steps") and output.exit_steps is not None:
                exit_steps = output.exit_steps
                if exit_steps.shape[1]>1:#说明是prefill阶段
                    exit_steps = exit_steps[:, -1]+1
                else:
                    exit_steps = exit_steps.squeeze(1)+1 #因为输出的步数是0,1,2,3，所以需要+1
                exit_steps_list.append(exit_steps)
        # 3. 注册钩子函数
        handle = self.model.register_forward_hook(exit_step_hook)
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=self.device.type,
                dtype=self.mixed_precision_dtype,
                enabled=self.mixed_precision_dtype is not None,
            ),
        ):
            # Ouro models work like causal LMs for text-only input
            ouro_output = self.model(inps, **self.ouro_config)
        handle.remove()
        # 4. 计算exit_steps_list的平均值，注意pad_token_id需要排除掉
        generation_length = inps.shape[0] #logprob默认每个数据就只生成一个token
        exit_steps_tensor = torch.stack(exit_steps_list).to(self.device) #(generation_length, batch_size)
        exit_steps_tensor = exit_steps_tensor.transpose(0, 1) #(batch_size, generation_length)
        return ouro_output.logits, generation_length, exit_steps_tensor.sum()
    
    def _model_generate(
        self,
        context,
        max_length: int,
        stop: list[str],
        **generation_kwargs,
    ) -> torch.Tensor:
        # temperature = 0.0 if not set
        # if do_sample is false and temp==0.0:
        # remove temperature, as do_sample=False takes care of this
        # and we don't want a warning from HF
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample")

        # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding strategies
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False

        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")
        # build stopping criteria
        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, context.shape[1], context.shape[0]
        )
        # 2. 定义专门提取 exit_steps 的钩子函数
        exit_steps_list = []
        def exit_step_hook(module, input, output):
            if hasattr(output, "exit_steps") and output.exit_steps is not None:
                exit_steps = output.exit_steps
                if exit_steps.shape[1]>1:#说明是prefill阶段
                    exit_steps = exit_steps[:, -1]+1
                else:
                    exit_steps = exit_steps.squeeze(1)+1 #因为输出的步数是0,1,2,3，所以需要+1
                exit_steps_list.append(exit_steps)
        # 3. 注册钩子函数
        handle = self.model.register_forward_hook(exit_step_hook)
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.mixed_precision_dtype,
            enabled=self.mixed_precision_dtype is not None,
        ):
            result = self.model.generate(
                input_ids=context,
                max_length=max_length,
                stopping_criteria=stopping_criteria,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
                **self.ouro_config,
                **generation_kwargs, #CHANGE一下
            )
        handle.remove()
        # 4. 计算exit_steps_list的平均值，注意pad_token_id需要排除掉
        generation_length = len(exit_steps_list)
        generation_mask = result[:,-generation_length:] != self.tokenizer.pad_token_id
        exit_steps_tensor = torch.stack(exit_steps_list).to(self.device) #(generation_length, batch_size)
        exit_steps_tensor = exit_steps_tensor.transpose(0, 1) #(batch_size, generation_length)
        exit_steps_tensor = exit_steps_tensor * generation_mask
        # exit_steps_seq_mean_token_mean = exit_steps_tensor.mean(dim=0).mean()
        # exit_steps_token_mean = sum(exit_steps_tensor)/(sum(generation_mask))
        
        return result, generation_mask.sum(), exit_steps_tensor.sum()
    def generate_until(
        self, requests: list[Instance], disable_tqdm: bool = False
    ) -> list[str]:
        res = []

        def _collate(req: tuple[str, dict]):
            """Defines the key for the sorted method"""
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(req[0])
            return -len(toks), req[0]

        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_until requests",
        )
        adaptive_batch_size = None
        if self.batch_size == "auto":
            # using rolling window with maximum context
            print("Passed argument batch_size = auto. Detecting largest batch size")
            batch_size = self._detect_batch_size()
            print(f"Determined Largest batch size: {batch_size}")
            adaptive_batch_size = batch_size
        # for each different set of kwargs, we execute all requests, by batch.
        batch_size = (
            self.batch_size
            if self.batch_size != "auto"
            else adaptive_batch_size
            if adaptive_batch_size is not None
            else 0
        )
        batch_fn = (
            self._batch_scheduler
            if self.batch_size == "auto" and not adaptive_batch_size
            else None
        )
        eval_logger.info(f"batch_size: {batch_size}")
        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        # group_fn=lambda x: x[1] -> x=(context, gen_kwargs)
        re_ords = Collator(
            [reg.args for reg in requests],
            sort_fn=_collate,
            group_by="gen_kwargs",
            group_fn=lambda x: x[1],
        )
        chunks = re_ords.get_batched(n=batch_size, batch_fn=batch_fn)
        eos = self.tok_decode(self.eot_token_id, skip_special_tokens=False)
        token_num_list = []
        exit_steps_list = []
        for chunk in chunks:
            contexts, all_gen_kwargs = zip(*chunk, strict=True)
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            # unpack our keyword arguments.
            if isinstance(gen_kwargs, dict):
                kwargs = copy.deepcopy(gen_kwargs)  # edge case for repeats > 1
                # add EOS token to stop sequences
                until = handle_stop_sequences(kwargs.pop("until", None), eos=eos)
            else:
                raise TypeError(
                    f"Expected `kwargs` to be of type `dict` but got {type(gen_kwargs)}"
                )
            if "max_gen_toks" in kwargs:
                max_gen_toks = kwargs.pop("max_gen_toks")
            else:
                max_gen_toks = self.max_gen_toks

            # set the max length in tokens of inputs ("context_enc")
            if self.backend == "causal":
                # max len for inputs = max length, minus room to generate the max new tokens
                max_ctx_len = self.max_length - max_gen_toks
                assert max_ctx_len > 0, (
                    f"Invalid configuration: requested max tokens to generate ({max_gen_toks}) must be less than model's maximum sequence length ({self.max_length})."
                )
            elif self.backend == "seq2seq":
                # max len for inputs = encoder's whole max_length
                max_ctx_len = self.max_length

            # encode, pad, and truncate contexts for this batch
            context_enc, attn_masks = self.tok_batch_encode(
                contexts,
                left_truncate_len=max_ctx_len,
                truncation=self.truncation,
            )
            context_enc = context_enc.to(self.device)
            attn_masks = attn_masks.to(self.device)

            if "max_length" not in kwargs:
                kwargs["max_length"] = context_enc.shape[1] + max_gen_toks
            #THREEGOLDCHANGE:1.add exit_steps
            # perform batched generation
            cont, token_num, exit_steps = self._model_generate(
                context=context_enc,
                attention_mask=attn_masks,
                stop=until,
                **kwargs,
            )
            token_num_list.append(token_num)
            exit_steps_list.append(exit_steps)
            cont_toks_list = cont.tolist()
            for cont_toks, context in zip(cont_toks_list, contexts, strict=True):
                # discard context + left-padding toks if using causal decoder-only LM
                if self.backend == "causal":
                    cont_toks = cont_toks[context_enc.shape[1] :]

                # Handle integer think_end_token: find last occurrence and strip tokens after it
                if isinstance(self.think_end_token, int):
                    think_token_indices = [
                        i
                        for i, token in enumerate(cont_toks)
                        if token == self.think_end_token
                    ]
                    if think_token_indices:
                        cont_toks = cont_toks[think_token_indices[-1] + 1 :]

                s = self.tok_decode(cont_toks)

                # Strip leading whitespace if we removed thinking tokens
                if isinstance(self.think_end_token, int):
                    s = s.lstrip()

                # Apply post-processing: remove stop sequences and string-based thinking tokens
                s = postprocess_generated_text(
                    generation=s,
                    stop=until,
                    think_end_token=self.think_end_token
                    if isinstance(self.think_end_token, str)
                    else None,
                )
                res.append(s)

                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), s)
                pbar.update(1)
        # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)
        
        pbar.close()
        #THREEGOLDCHANGE:2.sum and reduce
        token_num_sum = torch.tensor(token_num_list).sum().to(self.device)
        exit_steps_sum = torch.tensor(exit_steps_list).sum().to(self.device)
        eval_logger.info(f"rank: {self.rank}, token_num: {token_num_sum}, exit_steps: {exit_steps_sum}")
        if self.world_size > 1:
            token_num_sum = self.accelerator.reduce(token_num_sum, reduction="sum")
            exit_steps_sum = self.accelerator.reduce(exit_steps_sum, reduction="sum")
        else:
            token_num_sum = token_num_sum.item()
            exit_steps_sum = exit_steps_sum.item()
        if self.rank == 0:
            eval_logger.info(f"token_num_sum: {token_num_sum}, exit_steps_sum: {exit_steps_sum}, average_exit_steps: {exit_steps_sum/len(requests)}")
            if self.ouro_exit_step_output_dir is not None:
                import os
                import json
                from datetime import datetime
                data_id = datetime.now().isoformat().replace(":", "-")
                output_dir = os.path.join(self.ouro_exit_step_output_dir, self.model_name)
                os.makedirs(output_dir, exist_ok=True)
                with open(os.path.join(output_dir, f"result_exit_steps_{data_id}.json"), "w") as f:
                    json.dump({"token_num": token_num_sum.item(), "exit_steps": exit_steps_sum.item(), "average_exit_steps": exit_steps_sum.item()/token_num_sum.item()}, f)
                output_file_path = os.path.join(output_dir, f"result_exit_steps_{data_id}.json")
                eval_logger.info(f"save exit_steps to {output_file_path} successfully")
        self.accelerator.wait_for_everyone()
        return res
    def _loglikelihood_tokens(
        self,
        requests: list[tuple[tuple[str, str], list[int], list[int]]],
        disable_tqdm: bool = False,
        override_bs: int | None = None,
    ) -> list[tuple[float, bool]]:
        # TODO: implement some kind of efficient-request-middleware that lumps together requests with the same context
        res = []

        def _collate(req: tuple[tuple[str, str], list[int], list[int]]):
            """Defines the key for the sorted method."""
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end

            toks = req[1] + req[2]
            return -len(toks), tuple(toks)

        def _lookup_one_token_cont(req: tuple[tuple[str, str], list[int], list[int]]):
            """Defines the key to group and lookup one-token continuations."""
            # Use with group_by="contexts" (optional)"
            # allows for the creation of a lookup, so we can reuse logits in case of one-token continuations.
            # speeds up some multiple-choice tasks proportionally to the number of choices.
            # groups requests by context+continuation[:-1] and infer on one request/group.
            return req[-2] + req[-1][:-1]

        re_ord = Collator(
            requests,
            sort_fn=_collate,
            group_by="contexts"
            if self.backend == "causal" and self.logits_cache
            else None,
            group_fn=_lookup_one_token_cont,
        )

        # automatic (variable) batch size detection for vectorization
        # pull longest context sample from request
        n_reordered_requests = len(re_ord)
        batch_size = (
            self.batch_size
            if self.batch_size != "auto"
            else override_bs
            if override_bs is not None
            else 0
        )
        batch_fn = (
            self._batch_scheduler
            if self.batch_size == "auto"
            and n_reordered_requests > 0
            and not override_bs
            else None
        )

        chunks = re_ord.get_batched(n=batch_size, batch_fn=batch_fn)
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running loglikelihood requests",
        )
        #THREEGOLDCHANGE:1.add token_num and exit_steps
        token_num_list = []
        exit_steps_list = []
        for chunk in chunks:
            inps = []
            cont_toks_list = []
            inplens = []

            conts = []
            encoder_attns = []

            padding_len_inp = None
            padding_len_cont = None
            # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
            # tensors, then we pack them together into a batch, call the model, and then pick it all apart
            # again because vectorizing is annoying

            for _, context_enc, continuation_enc in chunk:
                # sanity check
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length

                # how this all works (illustrated on a causal decoder-only setup):
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # model  \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

                # when too long to fit in context, truncate from the left
                if self.backend == "causal":
                    total_length = len(context_enc) + len(continuation_enc)
                    if total_length > self.max_length + 1:
                        eval_logger.warning(
                            f"Combined length of context ({len(context_enc)}) and continuation ({len(continuation_enc)}) "
                            f"exceeds model's maximum length ({self.max_length}). "
                            f"Truncating {total_length - self.max_length + 1} tokens from the left."
                        )
                    inp = torch.tensor(
                        (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1],
                        dtype=torch.long,
                        device=self.device,
                    )
                    (inplen,) = inp.shape
                elif self.backend == "seq2seq":
                    inp = torch.tensor(
                        (context_enc)[-self.max_length :],
                        dtype=torch.long,
                        device=self.device,
                    )
                    (inplen,) = inp.shape

                    # build encoder attn masks
                    encoder_attns.append(torch.ones_like(inp))

                    cont = torch.tensor(
                        (continuation_enc)[-self.max_length :],
                        # TODO: left-shift these?
                        # TODO: our code assumes we never end up truncating conts for either model type
                        dtype=torch.long,
                        device=self.device,
                    )
                    (contlen,) = cont.shape

                    conts.append(cont)

                    padding_len_cont = (
                        max(padding_len_cont, contlen)
                        if padding_len_cont is not None
                        else contlen
                    )

                padding_len_inp = (
                    max(padding_len_inp, inplen)
                    if padding_len_inp is not None
                    else inplen
                )

                inps.append(inp)  # [1, inp_length]
                cont_toks_list.append(continuation_enc)
                inplens.append(inplen)

            # create encoder attn mask and batched conts, if seq2seq
            call_kwargs = {}
            if self.backend == "causal":
                batched_inps = pad_and_concat(
                    padding_len_inp, inps, padding_side="right"
                )  # [batch, padding_len_inp]
            elif self.backend == "seq2seq":
                # TODO: left-pad encoder inps and mask?
                batched_inps = pad_and_concat(
                    padding_len_inp, inps
                )  # [batch, padding_len_inp]
                batched_conts = pad_and_concat(
                    padding_len_cont, conts
                )  # [batch, padding_len_cont]
                batched_encoder_mask = pad_and_concat(
                    padding_len_inp, encoder_attns
                )  # [batch, padding_len_inp]
                call_kwargs = {
                    "attn_mask": batched_encoder_mask,
                    "labels": batched_conts,
                }
            #THREEGOLDCHANGE:2.add exit_steps
            multi_logits, token_num, exit_steps = self._model_call(batched_inps, **call_kwargs)
            multi_logits = F.log_softmax(
                multi_logits,
                dim=-1,
                dtype=self.softmax_dtype,
            )  # [batch, padding_length (inp or cont), vocab]
            token_num_list.append(token_num)
            exit_steps_list.append(exit_steps)
            for (request_str, ctx_tokens, _), logits, inplen, cont_toks in zip(
                chunk, multi_logits, inplens, cont_toks_list, strict=True
            ):
                # Slice to original seq length
                contlen = len(cont_toks)
                # take only logits in the continuation
                # (discard context toks if decoder-only ; discard right-padding)
                # also discards + checks for "virtual tokens" in the causal LM's input window
                # from prompt/prefix tuning tokens, if applicable
                ctx_len = (
                    inplen + (logits.shape[0] - padding_len_inp)
                    if self.backend == "causal"
                    else None
                )
                logits = self._select_cont_toks(logits, contlen=contlen, inplen=ctx_len)
                logits = logits.unsqueeze(0)  # [1, seq, vocab]

                # Check if per-token argmax is exactly equal to continuation
                greedy_tokens = logits.argmax(dim=-1)

                # check for one-token continuation cache hits.
                # noop in case group_by != "contexts" or no cache hit and returns the
                # original args. Otherwise, expands the logits batch dimension and yields each
                # batch along with matching continuation tokens and prompt strings.
                # logits -> [1, seq, vocab]
                for request_str, cont_toks, logits in re_ord.get_cache(  # noqa
                    req_str=request_str,
                    cxt_toks=ctx_tokens,
                    cont_toks=cont_toks,
                    logits=logits,
                ):
                    cont_toks = torch.tensor(
                        cont_toks, dtype=torch.long, device=self.device
                    ).unsqueeze(0)  # [1, seq]
                    # Use trailing slice [-cont_toks.shape[1]:] to handle variable length cont_len (but same ctx+cont[:-1]).
                    # i.e. continuations can be sliced at diff points. Collator ensures we have sufficient greedy_tokens
                    # by choosing key with longest cont if group_by="contexts".
                    max_equal = (
                        greedy_tokens[:, -cont_toks.shape[1] :] == cont_toks
                    ).all()

                    # Obtain log-probs at the corresponding continuation token indices
                    # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
                    logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(
                        -1
                    )  # [1, seq]

                    # Answer: (log prob, is-exact-match)
                    answer = (float(logits.sum()), bool(max_equal))

                    res.append(answer)

                    if request_str is not None:
                        # special case: loglikelihood_rolling produces a number of loglikelihood requests
                        # all with cache key None. instead do add_partial on the per-example level
                        # in the loglikelihood_rolling() function for those.
                        self.cache_hook.add_partial(
                            "loglikelihood", request_str, answer
                        )
                    pbar.update(1)

        pbar.close()
        #THREEGOLDCHANGE:2.sum and reduce
        token_num_sum = torch.tensor(token_num_list).sum().to(self.device)
        exit_steps_sum = torch.tensor(exit_steps_list).sum().to(self.device)
        eval_logger.info(f"rank: {self.rank}, token_num: {token_num_sum}, exit_steps: {exit_steps_sum}")
        if self.world_size > 1:
            token_num_sum = self.accelerator.reduce(token_num_sum, reduction="sum")
            exit_steps_sum = self.accelerator.reduce(exit_steps_sum, reduction="sum")
        else:
            token_num_sum = token_num_sum
            exit_steps_sum = exit_steps_sum
        if self.rank == 0:
            eval_logger.info(f"token_num_sum: {token_num_sum}, exit_steps_sum: {exit_steps_sum}, average_exit_steps: {exit_steps_sum/len(requests)}")
            if self.ouro_exit_step_output_dir is not None:
                import os
                import json
                from datetime import datetime
                data_id = datetime.now().isoformat().replace(":", "-")
                output_dir = os.path.join(self.ouro_exit_step_output_dir, self.model_name)
                os.makedirs(output_dir, exist_ok=True)
                with open(os.path.join(output_dir, f"result_exit_steps_{data_id}.json"), "w") as f:
                    json.dump({"token_num": token_num_sum.item(), "exit_steps": exit_steps_sum.item(), "average_exit_steps": exit_steps_sum.item()/token_num_sum.item()}, f)
                output_file_path = os.path.join(output_dir, f"result_exit_steps_{data_id}.json")
                eval_logger.info(f"save exit_steps to {output_file_path} successfully")
        if self.world_size > 1:
            self.accelerator.wait_for_everyone()
        return re_ord.get_original(res)
    @property
    def max_length(self) -> int:
        """Get the maximum sequence length for the model."""
        if self._max_length:
            return self._max_length

        seqlen_config_attrs = (
            "max_position_embeddings",
            "n_positions",
            "n_ctx",
        )

        # First check text_config if it exists (for VLM-style models like Ouro)
        if hasattr(self.model.config, "text_config"):
            text_config = self.model.config.text_config
            for attr in seqlen_config_attrs:
                if hasattr(text_config, attr):
                    return getattr(text_config, attr)

        # Fall back to main config
        for attr in seqlen_config_attrs:
            if hasattr(self.model.config, attr):
                return getattr(self.model.config, attr)

        # Check tokenizer
        if (
            hasattr(self.tokenizer, "model_max_length")
            and self.tokenizer.model_max_length < 1000000000
        ):
            return self.tokenizer.model_max_length

        return self._DEFAULT_MAX_LENGTH
