import os

# Set token chunk size to 256
os.environ["LMCACHE_CHUNK_SIZE"] = "256"
# Enable CPU memory backend
os.environ["LMCACHE_LOCAL_CPU"] = "True"
# Set CPU memory limit to 5GB
os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5.0"

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

# Configure KV cache transfer to use LMCache
ktc = KVTransferConfig(
    kv_connector="LMCacheConnectorV1",
    kv_role="kv_both",
)

# Initialize LLM with LMCache configuration
# Adjust gpu_memory_utilization based on your GPU memory
llm = LLM(model="Qwen/Qwen3-8B",
          kv_transfer_config=ktc,
          max_model_len=8000,
          gpu_memory_utilization=0.8)

# Create example prompts with shared prefix
shared_prompt = "Hello, how are you?" * 1000
prompts = [
    shared_prompt + "Hello, my name is",
]

# Define sampling parameters
sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=10)

# Run inference
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    generated_text = output.outputs[0].text
    print(f"Generated text: {generated_text!r}")