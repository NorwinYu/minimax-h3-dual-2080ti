"""MiniMax-H3 TP2 — tensor-parallel forward across two GPUs.

Insert between the patched model and the sampler:

    Loader -> FP16 Fix -> LoRA -> Chunk -> MODEL -> TP2 -> sampler

Unlike sequence parallelism this needs no second copy of the model: each block's
four linear weights are SHARDED (half per GPU, 5.5 GB each), which is what makes
it fit a 22 GB card. The norms and adaln stay replicated (they are tiny).

Register through the patcher (add_object_patch), not by assigning
block.forward directly: the FP16-Exact node owns
"diffusion_model.blocks.N.forward" for every block and ComfyUI re-applies the
patcher's object patches when the model is loaded, which would overwrite a bare
attribute assignment.
"""
import logging
import weakref

import torch

from .tp2 import phase_report, shard_block, tp2_block

LOG = "[H3 TP2]"
_CALLS = [0]
_STEP = {"t0": 0.0}
_HOOKED = [False]


def _install_model_management_hook():
    """Teach ComfyUI's model management about self-managed multi-device models.

    A model whose forward places its own weights (TP shards, sequence-parallel
    replicas) must not also be loaded onto one device by load_models_gpu: that
    load costs the full 11 GB and competes with the shards for bandwidth, which
    is exactly what made the parallel path slower than the baseline.

    The hook is scoped to patchers carrying `multi_device_managed = True`, so
    every other model keeps the stock behaviour.
    """
    if _HOOKED[0]:
        return
    import comfy.model_management as mm

    orig_load = mm.LoadedModel.model_load

    def model_load(self, lowvram_model_memory=0, force_patch_weights=False):
        if not getattr(self.model, "multi_device_managed", False):
            return orig_load(self, lowvram_model_memory, force_patch_weights)
        real_model = self.model.model
        self.real_model = weakref.ref(real_model)
        self.model_finalizer = weakref.finalize(real_model, mm.cleanup_models)
        self.model_finalizer.atexit = False
        return real_model

    orig_req = mm.LoadedModel.model_memory_required

    def model_memory_required(self, device):
        if getattr(self.model, "multi_device_managed", False):
            return 0
        return orig_req(self, device)

    mm.LoadedModel.model_load = model_load
    mm.LoadedModel.model_memory_required = model_memory_required
    _HOOKED[0] = True
    logging.info("%s installed model-management hook (self-managed multi-device "
                 "models skip load_models_gpu weight placement)", LOG)


def cuda_devices():
    return [torch.device("cuda:%d" % i) for i in range(torch.cuda.device_count())]


def _make_forward(sh, tag):
    def forward(x, t_emb, mod_segments, rope_freqs,
                transformer_options=None, attention=None, **kwargs):
        import time as _t
        if tag == 0:
            _CALLS[0] += 1
            if _CALLS[0] <= 2:
                logging.info("%s tp2 block ACTIVE x=%s dev=%s/%s", LOG, tuple(x.shape),
                             sh["dev0"], sh["dev1"])
            _STEP["t0"] = _t.perf_counter()
        out = tp2_block(sh, x, t_emb, mod_segments, rope_freqs,
                        transformer_options=transformer_options, attention=attention)
        if tag == 49:
            torch.cuda.synchronize()
            logging.info("%s 50-block loop wall time: %.1f s | phases/call(ms): %s",
                         LOG, _t.perf_counter() - _STEP["t0"], phase_report())
        return out
    forward._h3_tp2 = tag
    return forward


class MiniMaxH3TP2:
    @classmethod
    def INPUT_TYPES(cls):
        devs = [str(d) for d in cuda_devices()] or ["cuda:0"]
        return {
            "required": {
                "model": ("MODEL",),
                "replica_device": (devs, {"default": devs[-1]}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "MiniMaxH3"
    DESCRIPTION = (
        "Run the MiniMax-H3 DiT tensor-parallel across two GPUs: every linear "
        "layer is sharded (half per GPU) with an NVLink allreduce, so the weights "
        "need only ~5.5 GB per card. Norms/adaln stay replicated."
    )

    def apply(self, model, replica_device):
        import comfy.model_management as mm

        _install_model_management_hook()
        dm = model.model.diffusion_model
        dev0 = torch.device(str(getattr(model, "load_device", None) or mm.get_torch_device()))
        dev1 = torch.device(replica_device)
        if dev0 == dev1:
            raise RuntimeError(f"{LOG} replica_device must differ from the model device {dev0}")

        # LoRA lives in the patcher's weight-patches and is applied at load time,
        # which is AFTER this node runs -- so slicing the raw block weights would
        # silently drop it. Extract the low-rank factors here and shard them too.
        lp = getattr(model, "patches", {}) or {}
        for i, blk in enumerate(dm.blocks):
            prefix = "diffusion_model.blocks.%d." % i
            lora = {}
            for name, sub in (("qkv", "attn.qkv_proj"), ("op", "attn.out_proj"),
                              ("fc1", "mlp.fc1"), ("fc2", "mlp.fc2")):
                ps = lp.get(prefix + sub + ".weight")
                if not ps:
                    continue
                sp, ad, sm, _off, _fn = ps[0]
                w = getattr(ad, "weights", None)
                if w is None or getattr(w[0], "ndim", 0) != 2:
                    continue
                rank = w[1].shape[0]
                lora[name] = (w[0].to(torch.float16), w[1].to(torch.float16),
                              float(w[2]) / rank * float(sp) * float(sm))
            sh = shard_block(blk, dev0, dev1, lora)
            fwd = _make_forward(sh, i)
            model.add_object_patch("diffusion_model.blocks.%d.forward" % i, fwd)
            blk.forward = fwd

        # the forward places every weight itself; stop load_models_gpu from also
        # loading the full model onto one device
        model.multi_device_managed = True
        # and tell the model's own forward to skip aimdo's single-device malloc
        # graph, which it would otherwise break by allocating on the second GPU
        dm.multi_device_managed = True

        q = dm.blocks[0].attn.qkv_proj
        ps = getattr(model, "patches", {}).get("diffusion_model.blocks.0.attn.qkv_proj.weight")
        if ps:
            sp, ad, sm, off, fn = ps[0]
            w = getattr(ad, "weights", None)
            logging.info("%s lora dump: n=%d strength_patch=%s strength_model=%s "
                         "len(weights)=%s up=%s down=%s alpha=%s", LOG, len(ps), sp, sm,
                         len(w) if w else None,
                         tuple(w[0].shape) if w else None,
                         tuple(w[1].shape) if w else None,
                         w[2] if w else None)
        logging.info(
            "%s patched %d blocks: primary=%s replica=%s | model.patches=%d "
            "| block0 qkv weight_function=%s | self_managed=True",
            LOG, len(dm.blocks), dev0, dev1, len(getattr(model, "patches", {})),
            getattr(q, "weight_function", None) is not None)
        return (model,)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3TP2": MiniMaxH3TP2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3TP2": "MiniMax-H3 TP2 (2-GPU tensor parallel)",
}
