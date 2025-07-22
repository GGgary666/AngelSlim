# Copyright 2025 Tencent Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os

from angelslim.engine import Engine
from angelslim.utils import get_yaml_prefix_simple
from angelslim.utils.config_parser import SlimConfigParser, print_config


def get_args():
    parser = argparse.ArgumentParser(description="AngelSlim")
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--save-path", type=str, default=None)
    args = parser.parse_args()
    return args


def merge_config(config, args):
    """
    Merge command line arguments into the configuration dictionary.

    Args:
        config (dict): Configuration dictionary to be updated.
        args (argparse.Namespace): Parsed command line arguments.
    """
    if args.save_path is not None:
        config.global_config.save_path = args.save_path
    if args.model_path is not None:
        config.model_config.model_path = args.model_path
    config.global_config.save_path = os.path.join(
        config.global_config.save_path,
        get_yaml_prefix_simple(args.config),
    )


def run(config):
    """
    Run the LLM compression process based on the provided configuration.

    Args:
        config (dict): Configuration dictionary containing
                       parameters for LLM compression.
    """
    # Step 1: Initialize configurations
    model_config = config.model_config
    dataset_config = config.dataset_config
    compress_config = config.compression_config
    global_config = config.global_config

    # Step 2: Execute complete pipeline
    slim_engine = Engine()

    # Step 3: Prepare model
    slim_engine.prepare_model(
        model_name=model_config.name,
        model_path=model_config.model_path,
        torch_dtype=model_config.torch_dtype,
        device_map=model_config.device_map,
        trust_remote_code=model_config.trust_remote_code,
        low_cpu_mem_usage=model_config.low_cpu_mem_usage,
        use_cache=model_config.use_cache,
        deploy_backend=global_config.deploy_backend,
    )

    # Step 4: Prepare data (optional custom dataloader)
    if compress_config.need_dataset:
        slim_engine.prepare_data(
            data_path=dataset_config.data_path,
            data_type=dataset_config.name,
            custom_dataloader=None,
            max_length=dataset_config.max_seq_length,
            batch_size=dataset_config.batch_size,
            num_samples=dataset_config.num_samples,
            shuffle=dataset_config.shuffle,
        )

    # Step 5: Initialize compressor
    slim_engine.prepare_compressor(
        compress_name=compress_config.name,
        compress_config=compress_config,
        global_config=global_config,
    )

    # Step 6: Compress model
    slim_engine.run()

    # Step 7: Save compressed model
    slim_engine.save(global_config.save_path, config)


if __name__ == "__main__":
    args = get_args()
    parser = SlimConfigParser()
    config = parser.parse(args.config)
    merge_config(config, args)
    print_config(config)
    run(config)
