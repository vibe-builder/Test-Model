"""Lightweight profiling helpers for Nano XYZ."""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import dataclass, asdict
from typing import Iterable, Tuple

import torch

from nano_xyz.configuration_nano import NanoConfig
from nano_xyz.modeling_nano import NanoForCausalLM


@dataclass
class BenchmarkResult:
    name: str
    seq_len: int
    tokens_per_second: float
    peak_memory_mb: float
    run_index: int
    torch_version: str
    device: str
    device_name: str


def _profile_once(config: NanoConfig, seq_len: int, device: torch.device, run_idx: int) -> BenchmarkResult:
    model = NanoForCausalLM(config)
    model.to(device)
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (1, seq_len), device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()
    with torch.inference_mode():
        model(input_ids=input_ids)
    elapsed = max(time.perf_counter() - start, 1e-6)

    tokens_per_second = seq_len / elapsed
    peak_mem = (
        torch.cuda.max_memory_allocated(device) / (1024**2)
        if device.type == "cuda"
        else 0.0
    )

    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()
    return BenchmarkResult(
        name=f"{config.n_layer}L_{config.block_size}",
        seq_len=seq_len,
        tokens_per_second=tokens_per_second,
        peak_memory_mb=peak_mem,
        run_index=run_idx,
        torch_version=torch.__version__,
        device=device.type,
        device_name=device_name,
    )


def run_benchmarks(
    configs: Iterable[Tuple[str, NanoConfig]],
    seq_lens: Iterable[int],
    repeat: int,
    log_path: str | None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Nano benchmarks on {device}.")

    log_file = open(log_path, "w", encoding="utf-8") if log_path else None
    results: list[BenchmarkResult] = []
    for label, config in configs:
        for seq in seq_lens:
            for run_idx in range(repeat):
                cfg = config.__class__(**config.to_dict())
                cfg.block_size = max(cfg.block_size, seq)
                result = _profile_once(cfg, seq, device, run_idx)
                result.name = label
                results.append(result)
                if log_file:
                    log_file.write(json.dumps(asdict(result)) + "\n")

    for result in results:
        mem = f"{result.peak_memory_mb:,.1f} MB" if result.peak_memory_mb else "N/A"
        print(
            f"{result.name:<15} | seq={result.seq_len:<5} | "
            f"{result.tokens_per_second:8.2f} tok/s | peak mem: {mem} | run={result.run_index}"
        )

    if log_file:
        log_file.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile Nano XYZ throughput.")
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--repeat", type=int, default=3, help="Number of runs per configuration.")
    parser.add_argument("--log-json", type=str, default=None, help="Optional path to JSONL output.")
    parser.add_argument("--yarn-targets", type=int, nargs="*", default=[4096, 8192], help="YaRN target contexts.")
    return parser.parse_args()


def build_default_configs(yarn_targets: Iterable[int]) -> list[Tuple[str, NanoConfig]]:
    base = NanoConfig(
        n_layer=4,
        n_head=4,
        n_embd=256,
        use_lcr=False,
        use_gtr=False,
    )
    configs: list[Tuple[str, NanoConfig]] = [("vanilla", base)]
    for target in yarn_targets:
        configs.append(
            (
                f"yarn_{target}",
                NanoConfig(
                    n_layer=4,
                    n_head=4,
                    n_embd=256,
                    use_yarn=True,
                    yarn_target_ctx=target,
                    use_lcr=False,
                    use_gtr=False,
                ),
            )
        )
    hybrid = NanoConfig(
        n_layer=4,
        n_head=4,
        n_embd=256,
        use_lcr=True,
        lcr_block_indices=[1],
        use_gtr=True,
        gtr_block_indices=[3],
    )
    configs.append(("hybrid", hybrid))
    return configs


def main() -> None:
    args = parse_args()
    configs = build_default_configs(args.yarn_targets)
    run_benchmarks(configs, args.seq_lens, args.repeat, args.log_json)


if __name__ == "__main__":
    main()
