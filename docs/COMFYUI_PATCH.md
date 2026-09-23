# Required one-line ComfyUI edit (aimdo opt-out)

ComfyUI's `aimdo` prefetch records a **single-device** allocation graph around the
model forward. A tensor-parallel model places its own weights across several GPUs,
so its forward allocates on other devices too — which breaks that graph with

```
RuntimeError: aimdo memory compile error
```

The TP2 node flags the model with `multi_device_managed`, and the model forward
opts out of aimdo when that flag is set. This is a **one-line change to ComfyUI
core**, which this repository does **not** redistribute. Make it yourself:

File: `comfy/ldm/minimax/model.py` (in `MiniMaxH3Model.forward`)

Before:

```python
compile_allocations = comfy.model_prefetch.malloc_graph_enabled(x[0].device)
```

After:

```python
# aimdo records its allocation graph for a single device/stream. A model whose
# forward places weights itself across several devices (TP / sequence parallel)
# allocates on other devices too, which breaks that graph, so such a model opts out.
compile_allocations = comfy.model_prefetch.malloc_graph_enabled(x[0].device) \
    and not getattr(self, "multi_device_managed", False)
```

That is the only ComfyUI-core change the whole setup needs. Everything else lives
in `custom_nodes/comfyui-h3-tp2/`.
