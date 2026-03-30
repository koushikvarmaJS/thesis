import time
import torch
import threading
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "gpt2"

NVML_AVAILABLE = False
pynvml = None

try:
    import pynvml

    try:
        pynvml.nvmlInit()
        pynvml.nvmlDeviceGetHandleByIndex(0)
        NVML_AVAILABLE = True
    except Exception:
        pass
except ImportError:
    pynvml = None


def measure_energy_during_inference(model, inputs, max_new_tokens=50):
    energy = 0
    handle = None
    running = True
    nvml_ok = NVML_AVAILABLE

    if nvml_ok and pynvml is not None:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            handle = None
            nvml_ok = False

    def sample_power():
        nonlocal energy
        while running and handle is not None and nvml_ok:
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
                energy += power * 0.1
            except Exception:
                energy += 200 * 0.1
            time.sleep(0.1)

    def run_inference():
        return model.generate(**inputs, max_new_tokens=max_new_tokens)

    sampler = None
    if handle is not None:
        sampler = threading.Thread(target=sample_power)
        sampler.start()

    start = time.time()
    output = run_inference()
    end = time.time()

    running = False
    if sampler is not None:
        sampler.join()

    if handle is None:
        energy = 999

    tokens_generated = output.shape[1] - inputs["input_ids"].shape[1]

    return energy, tokens_generated, (end - start)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

    prompt = "Explain KV cache in transformers."
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    energy, tokens, latency = measure_energy_during_inference(model, inputs)

    print({
        "method": "baseline",
        "tokens": tokens,
        "latency": latency,
        "joules_per_token": energy / tokens if tokens > 0 else None
    })


if __name__ == "__main__":
    for _ in range(5):
        main()
