#!/usr/bin/env python3
"""
Comprehensive performance benchmarking for Nano XYZ models.

Benchmarks:
- Inference latency across different sequence lengths
- Memory usage patterns
- DCA vs Dense attention comparison
- Throughput measurements
- Quantization speedup validation

Usage:
    python benchmark_performance.py --model-size tiny --attention-type dca
    python benchmark_performance.py --model-size medium --compare-attention
"""

import argparse
import json
import logging
import time
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from nano_xyz import NanoModel, NanoConfig, NanoEncoderDecoderModelForSeq2SeqLM
except ImportError as e:
    logger.error(f"Failed to import nano_xyz: {e}")
    sys.exit(1)


@contextmanager
def cuda_timer():
    """CUDA event-based timer for accurate GPU timing."""
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        yield lambda: start_event.elapsed_time(end_event)
        end_event.record()
    else:
        start_time = time.time()
        yield lambda: (time.time() - start_time) * 1000  # Convert to ms


def measure_memory_usage():
    """Measure current GPU memory usage."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**3)  # GB
    return 0.0


def benchmark_inference_latency(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    num_runs: int = 10,
    warmup_runs: int = 3
) -> Dict[str, float]:
    """
    Benchmark inference latency with proper warmup and averaging.

    Args:
        model: Model to benchmark
        input_ids: Input token ids
        attention_mask: Optional attention mask
        num_runs: Number of measurement runs
        warmup_runs: Number of warmup runs

    Returns:
        Dictionary with latency statistics
    """
    model.eval()
    device = next(model.parameters()).device

    # Move inputs to device
    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # Warmup
    logger.info(f"Running {warmup_runs} warmup iterations...")
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_ids, attention_mask=attention_mask)

    # Synchronize before measurement
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Measure latency
    latencies = []
    logger.info(f"Running {num_runs} measurement iterations...")

    with torch.no_grad():
        for _ in range(num_runs):
            with cuda_timer() as timer:
                _ = model(input_ids, attention_mask=attention_mask)
            latencies.append(timer())

    # Calculate statistics
    latencies = torch.tensor(latencies)
    return {
        "mean_latency_ms": latencies.mean().item(),
        "std_latency_ms": latencies.std().item(),
        "min_latency_ms": latencies.min().item(),
        "max_latency_ms": latencies.max().item(),
        "median_latency_ms": latencies.median().item(),
        "throughput_tokens_per_sec": (input_ids.numel() * 1000) / latencies.mean().item(),
    }


def benchmark_memory_usage(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Benchmark memory usage during inference.

    Args:
        model: Model to benchmark
        input_ids: Input token ids
        attention_mask: Optional attention mask

    Returns:
        Dictionary with memory statistics
    """
    model.eval()
    device = next(model.parameters()).device

    # Move inputs to device
    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # Clear cache and measure baseline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    baseline_memory = measure_memory_usage()

    with torch.no_grad():
        _ = model(input_ids, attention_mask=attention_mask)

    peak_memory = measure_memory_usage()
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / (1024**3)

    return {
        "baseline_memory_gb": baseline_memory,
        "peak_memory_gb": peak_memory,
        "memory_increase_gb": peak_memory - baseline_memory,
        "memory_per_token_gb": (peak_memory - baseline_memory) / input_ids.numel(),
    }


def benchmark_attention_comparison(
    seq_len: int,
    batch_size: int = 1,
    vocab_size: int = 10000
) -> Dict[str, Any]:
    """
    Compare DCA vs Dense attention performance.

    Args:
        seq_len: Sequence length to test
        batch_size: Batch size
        vocab_size: Vocabulary size for input generation

    Returns:
        Comparison results
    """
    logger.info(f"Benchmarking DCA vs Dense attention for seq_len={seq_len}, batch_size={batch_size}")

    # Create DCA config
    dca_config = NanoConfig(
        vocab_size=vocab_size,
        n_layer=6,
        n_head=8,
        n_embd=512,
        block_size=max(seq_len * 2, 1024),  # Ensure it can handle the sequence
        attention_type="dca",
        dca_local_window=min(512, seq_len // 4),
        dca_global_tokens=min(64, seq_len // 16),
    )

    # Create Dense config
    dense_config = NanoConfig(**dca_config.__dict__)
    dense_config.attention_type = "default"

    # Create models
    dca_model = NanoModel(dca_config)
    dense_model = NanoModel(dense_config)

    # Generate input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Benchmark DCA
    logger.info("Benchmarking DCA attention...")
    dca_latency = benchmark_inference_latency(dca_model, input_ids)
    dca_memory = benchmark_memory_usage(dca_model, input_ids)

    # Benchmark Dense (may fail for very long sequences)
    dense_results = {}
    try:
        logger.info("Benchmarking Dense attention...")
        dense_latency = benchmark_inference_latency(dense_model, input_ids)
        dense_memory = benchmark_memory_usage(dense_model, input_ids)
        dense_results = {
            "latency": dense_latency,
            "memory": dense_memory,
        }
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.warning(f"Dense attention failed with OOM for seq_len={seq_len}")
            dense_results = {"error": "out_of_memory"}
        else:
            raise

    return {
        "sequence_length": seq_len,
        "batch_size": batch_size,
        "dca": {
            "latency": dca_latency,
            "memory": dca_memory,
        },
        "dense": dense_results,
        "speedup": dca_latency["mean_latency_ms"] / dense_results.get("latency", {}).get("mean_latency_ms", float('inf'))
        if dense_results and "latency" in dense_results else None,
    }


def benchmark_quantization_speedup(
    seq_len: int,
    batch_size: int = 1,
    vocab_size: int = 10000
) -> Dict[str, Any]:
    """
    Benchmark quantization speedup.

    Args:
        seq_len: Sequence length to test
        batch_size: Batch size
        vocab_size: Vocabulary size

    Returns:
        Quantization comparison results
    """
    logger.info(f"Benchmarking quantization speedup for seq_len={seq_len}, batch_size={batch_size}")

    # Base config
    config = NanoConfig(
        vocab_size=vocab_size,
        n_layer=6,
        n_head=8,
        n_embd=512,
        block_size=max(seq_len * 2, 1024),
        attention_type="dca",
    )

    # FP32 model
    fp32_config = NanoConfig(**config.__dict__)
    fp32_model = NanoModel(fp32_config)

    # Quantized model
    quant_config = NanoConfig(**config.__dict__)
    quant_config.quantization_config = {
        "method": "torchao",
        "quant_type": "int8_dyn_act_int4_weight",
    }
    quant_model = NanoModel(quant_config)

    # Generate input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Benchmark both
    fp32_latency = benchmark_inference_latency(fp32_model, input_ids)
    quant_latency = benchmark_inference_latency(quant_model, input_ids)

    return {
        "sequence_length": seq_len,
        "batch_size": batch_size,
        "fp32_latency": fp32_latency,
        "quantized_latency": quant_latency,
        "speedup": fp32_latency["mean_latency_ms"] / quant_latency["mean_latency_ms"],
    }


def run_comprehensive_benchmark(
    model_sizes: List[str] = ["tiny", "small"],
    seq_lengths: List[int] = [512, 1024, 2048, 4096],
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run comprehensive benchmark suite.

    Args:
        model_sizes: List of model sizes to test
        seq_lengths: List of sequence lengths to test
        output_file: Optional file to save results

    Returns:
        Benchmark results dictionary
    """
    results = {
        "timestamp": time.time(),
        "model_sizes": model_sizes,
        "sequence_lengths": seq_lengths,
        "results": {},
    }

    for model_size in model_sizes:
        logger.info(f"Benchmarking model size: {model_size}")
        results["results"][model_size] = {}

        # Load config
        config = NanoConfig.from_preset(f"decoder_{model_size}")

        # Test different sequence lengths
        for seq_len in seq_lengths:
            logger.info(f"Testing sequence length: {seq_len}")

            # Create model
            model = NanoModel(config)

            # Generate input
            batch_size = 1
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

            # Run benchmarks
            latency_results = benchmark_inference_latency(model, input_ids)
            memory_results = benchmark_memory_usage(model, input_ids)

            results["results"][model_size][seq_len] = {
                "latency": latency_results,
                "memory": memory_results,
            }

    # Save results if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Nano XYZ Performance Benchmarking")
    parser.add_argument("--model-size", choices=["tiny", "small", "medium"], default="tiny")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--compare-attention", action="store_true", help="Compare DCA vs Dense attention")
    parser.add_argument("--benchmark-quantization", action="store_true", help="Benchmark quantization speedup")
    parser.add_argument("--comprehensive", action="store_true", help="Run comprehensive benchmark suite")
    parser.add_argument("--output-file", type=str, help="Save results to JSON file")

    args = parser.parse_args()

    if args.comprehensive:
        results = run_comprehensive_benchmark(
            model_sizes=["tiny", "small"],
            seq_lengths=[512, 1024, 2048, 4096],
            output_file=args.output_file
        )
        print(json.dumps(results, indent=2, default=str))
        return

    if args.compare_attention:
        results = benchmark_attention_comparison(args.seq_len, args.batch_size)
        print(json.dumps(results, indent=2, default=str))
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        return

    if args.benchmark_quantization:
        results = benchmark_quantization_speedup(args.seq_len, args.batch_size)
        print(json.dumps(results, indent=2, default=str))
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        return

    # Default: benchmark single model
    config = NanoConfig.from_preset(f"decoder_{args.model_size}")
    model = NanoModel(config)

    input_ids = torch.randint(0, config.vocab_size, (args.batch_size, args.seq_len))

    latency_results = benchmark_inference_latency(model, input_ids)
    memory_results = benchmark_memory_usage(model, input_ids)

    results = {
        "model_size": args.model_size,
        "sequence_length": args.seq_len,
        "batch_size": args.batch_size,
        "latency": latency_results,
        "memory": memory_results,
    }

    print(json.dumps(results, indent=2, default=str))
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()

