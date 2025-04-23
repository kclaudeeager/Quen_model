import os
import time
import json
import asyncio
import argparse
from typing import Optional, List, Dict, Any
import pandas as pd
import jsonlines
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import aiohttp
from functools import lru_cache
import hashlib

# Try to import vLLM for local model support
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

try:
    from openai import AsyncOpenAI
    ASYNC_OPENAI_AVAILABLE = True
except ImportError:
    ASYNC_OPENAI_AVAILABLE = False
    from openai import OpenAI

# Default settings for Evaluator
MAX_TOKENS = 4096
OPENAI_MODEL = 'gpt-4-1106-preview'
TEMPERATURE = 0.7
TOP_P: float = 1.0
NUM_RETURN_SEQUENCES: int = 1
BATCH_SIZE: int = 10  # Process this many examples at once
MAX_CONCURRENT_REQUESTS: int = 5  # Maximum concurrent API calls
CACHE_DIR = ".cache"  # Directory to cache evaluation results

PROMPT_TYPE_DICT = {
    'gsm8k': 'gsm8k_auto',
    'strategyqa': 'sq_auto',
    'aqua': 'aqua_auto',
    'cosmos': 'cosmos_auto',
    'multistep_arithmetic': 'arith_auto',
    'word_sorting': 'sort_auto',
    'logical_deduction': 'logic_auto'
}

def load_api_key():
    """Load OpenAI API key from environment variable or .env file"""
    try:
        from dotenv import load_dotenv
        load_dotenv()  # Load from .env file if exists
    except ImportError:
        pass  # dotenv not installed
        
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError(
            "OpenAI API key not found. Please set OPENAI_API_KEY environment variable "
            "or create a .env file with OPENAI_API_KEY=your_key"
        )
    return api_key

# Create cache directory if it doesn't exist
os.makedirs(CACHE_DIR, exist_ok=True)

@lru_cache(maxsize=1)
def get_client():
    """Get cached API client"""
    if ASYNC_OPENAI_AVAILABLE:
        return AsyncOpenAI(api_key=load_api_key())
    else:
        return OpenAI(api_key=load_api_key())

@lru_cache(maxsize=1)
def get_vllm_model(model_path):
    """Load and cache vLLM model"""
    if not VLLM_AVAILABLE:
        raise ImportError("vLLM is not installed. Please install it with 'pip install vllm'")
    
    return LLM(
        model=model_path,
        tensor_parallel_size=1,  # Adjust based on your GPU setup
        trust_remote_code=True,
        dtype="float16",
        gpu_memory_utilization=0.85,
        max_model_len=2048
    )

def get_cache_key(prompt: str) -> str:
    """Generate a cache key from the prompt"""
    return hashlib.md5(prompt.encode()).hexdigest()

def check_cache(cache_key: str) -> Optional[List[str]]:
    """Check if result is in cache"""
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)
    return None

def save_to_cache(cache_key: str, result: List[str]):
    """Save result to cache"""
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    with open(cache_file, 'w') as f:
        json.dump(result, f)

async def generate_async(prompts: List[str], use_local_model: bool = False, local_model_path: Optional[str] = None):
    """Generate completions asynchronously using OpenAI API"""
    results = []
    client = get_client()
    
    # Check cache first for each prompt
    cached_results = []
    prompts_to_process = []
    cache_keys = []
    
    for prompt in prompts:
        cache_key = get_cache_key(prompt)
        cached_result = check_cache(cache_key)
        if cached_result:
            cached_results.append((True, cached_result))
        else:
            cached_results.append((False, None))
            prompts_to_process.append(prompt)
            cache_keys.append(cache_key)
    
    # Process prompts not in cache
    if prompts_to_process:
        if use_local_model and VLLM_AVAILABLE:
            # Use vLLM for local model inference
            model = get_vllm_model(local_model_path)
            outputs = model.generate(
                prompts_to_process,
                SamplingParams(
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    max_tokens=MAX_TOKENS,
                    n=NUM_RETURN_SEQUENCES,
                )
            )
            api_results = [[output.outputs[0].text] for output in sorted(outputs, key=lambda x: int(x.request_id))]
        else:
            # Use OpenAI API with concurrency limit
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
            
            async def process_prompt(prompt):
                async with semaphore:
                    try:
                        messages = [{'role': 'user', 'content': prompt}]
                        response = await client.chat.completions.create(
                            model=OPENAI_MODEL,
                            messages=messages,
                            max_tokens=MAX_TOKENS,
                            temperature=TEMPERATURE,
                            top_p=TOP_P,
                            n=NUM_RETURN_SEQUENCES,
                        )
                        return [choice.message.content for choice in response.choices]
                    except Exception as e:
                        print(f'An Error Occurred: {e}, retrying in 5 seconds')
                        await asyncio.sleep(5)
                        return await process_prompt(prompt)  # Retry once
            
            # Process all prompts concurrently
            tasks = [process_prompt(prompt) for prompt in prompts_to_process]
            api_results = await asyncio.gather(*tasks)
            
            # Cache the results
            for i, result in enumerate(api_results):
                save_to_cache(cache_keys[i], result)
    else:
        api_results = []
    
    # Combine cached results and new results
    result_idx = 0
    for is_cached, cached_result in cached_results:
        if is_cached:
            results.append(cached_result)
        else:
            results.append(api_results[result_idx])
            result_idx += 1
            
    return results

def generate_sync(prompts: List[str], use_local_model: bool = False, local_model_path: Optional[str] = None):
    """Synchronous version of generate for environments that don't support asyncio"""
    results = []
    client = get_client()
    
    for prompt in prompts:
        cache_key = get_cache_key(prompt)
        cached_result = check_cache(cache_key)
        
        if cached_result:
            results.append(cached_result)
            continue
            
        if use_local_model and VLLM_AVAILABLE:
            model = get_vllm_model(local_model_path)
            output = model.generate(
                [prompt],
                SamplingParams(
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    max_tokens=MAX_TOKENS,
                    n=NUM_RETURN_SEQUENCES,
                )
            )
            text = [output[0].outputs[0].text]
        else:
            messages = [{'role': 'user', 'content': prompt}]
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                n=NUM_RETURN_SEQUENCES,
            )
            text = [choice.message.content for choice in response.choices]
            
        save_to_cache(cache_key, text)
        results.append(text)
        
    return results

def autorace_criterion(dataset:str = 'aqua', example_wrong_chains:str = 'EXAMPLE_WRONG_CHAINS_AQUA.txt'):
    """Generate criterions by comparing reference/student answers."""
    assert os.path.exists(example_wrong_chains), f'example_wrong_chains: {example_wrong_chains} does not exist!'
    
    with open(example_wrong_chains) as f:
        EXAMPLE_WRONG_CHAINS = f.read()

    with open('prompt.json') as f:
        prompt = json.load(f)
        
    if f"{dataset}_auto" in prompt:
        print(f'Warning: dataset {dataset} already exists in prompt.json, please check whether you want to overwrite it.')
        input('Press any key to continue...')
        
    criterion_prompt = prompt['criterion'].format(EXAMPLE_WRONG_CHAINS)
    
    # Use sync version for this case
    criterion_text = generate_sync([criterion_prompt])[0]
    
    print(criterion_text)
    criterion = '1. **' + criterion_text[0].split('1. **')[-1]
    
    import re
    criterion = re.sub(r'\d\. ', '', criterion)
    evaluation_prompt = 'Below is a question and an answer from a student. You are required to check the correctness of the reasoning chains step by step. The criterions are as follows:\n\n{}\n\nQuestion:\n{{}}\n\nStudent answer:\n{{}}\n\nPlease check the answer through each criterion, and make sure you carefully examine each reasoning step. Finally, if there is any step that fails the verification, output a INCORRECT, else output a CORRECT.'.format(criterion)
    prompt[dataset + '_auto'] = evaluation_prompt

    with open('prompt.json', 'w') as f:
        json.dump(prompt, f)

def autorace_score(output_log_path:str):
    """Report autorace score"""
    with jsonlines.open(output_log_path, mode='r') as reader:
        autorace = list(reader)

    total = len(autorace)
    incorrect = 0
    for i in range(total):
        if 'INCORRECT' in autorace[i]['evaluation_result'][0]:
            incorrect += 1

    print(f'autorace score: {(total - incorrect) / total:.2f}')

async def autorace_evaluation_async(
    dataset: str = "gsm8k", 
    reasoning_model: str = "eval_model",
    output_log_dir: str = 'logs/auto_race',
    use_local_model: bool = False,
    local_model_path: Optional[str] = None,
    batch_size: int = BATCH_SIZE
):
    """Asynchronous version of autorace evaluation"""
    predefined_datasets = ['gsm8k', 'strategyqa', 'aqua', 'cosmos', 'multistep_arithmetic', 'word_sorting', 'logical_deduction']
    
    if dataset not in predefined_datasets:
        print(f"Warning: The dataset '{dataset}' is not a predefined dataset.")
    if dataset not in PROMPT_TYPE_DICT:
        raise ValueError(f"dataset '{dataset}' is not in PROMPT_TYPE_DICT! Please add the prompt type to PROMPT_TYPE_DICT.")
    
    data_path = f'./data/{reasoning_model}/{dataset}.jsonl'
    assert os.path.exists(data_path), f'the output from {reasoning_model}: {data_path} does not exist!'
    
    output_log_dir = os.path.join(output_log_dir, reasoning_model, dataset)
    os.makedirs(output_log_dir, exist_ok=True)
    output_log_path = f'{output_log_dir}/autorace_eval.jsonl'
    
    print(f"Evaluating reasoning model: {reasoning_model} on dataset: {dataset}, output log path: {output_log_path}")
    
    # Load data
    data = pd.read_json(data_path, lines=True)
    
    # Load prompts
    with open('prompt.json') as f:
        prompts = json.load(f)
    
    # Process in batches
    results = []
    sample_batches = []
    prompt_batches = []
    
    # Format and prepare all examples for batched processing
    formatted_examples = []
    for index in range(len(data)):
        reasoning_chain = data.loc[index, 'reasoning_chain']
        if not reasoning_chain.startswith('\n'):
            reasoning_chain = '\n' + reasoning_chain
        reasoning_chain = reasoning_chain.rstrip('\n\n.')
        
        raw_question = data.loc[index, 'question']
        raw_question = raw_question.replace('Q:', '')
        raw_question = raw_question.lstrip(' ')
        
        prompt = prompts[PROMPT_TYPE_DICT[dataset]].format(raw_question, reasoning_chain)
        prompt = prompt.replace('..', '.')
        
        formatted_examples.append({
            'index': index,
            'question': raw_question,
            'reasoning_chain': reasoning_chain,
            'prompt': prompt,
            'answer': data.loc[index, 'answer']
        })
    
    # Create batches
    for i in range(0, len(formatted_examples), batch_size):
        batch = formatted_examples[i:i+batch_size]
        sample_batches.append(batch)
        prompt_batches.append([example['prompt'] for example in batch])
    
    # Process all batches
    total_batches = len(prompt_batches)
    for batch_idx in tqdm(range(total_batches), desc="Processing batches"):
        sample_batch = sample_batches[batch_idx]
        prompt_batch = prompt_batches[batch_idx]
        
        evaluation_results = await generate_async(prompt_batch, use_local_model, local_model_path)
        
        for i, sample in enumerate(sample_batch):
            result = {
                'index': sample['index'],
                'evaluation_result': evaluation_results[i],
                'question': sample['question'],
                'reasoning_chain': sample['reasoning_chain'],
                'answer': sample['answer'],
                'prompt': sample['prompt']
            }
            results.append(result)
            
        # Save after each batch
        with jsonlines.open(output_log_path, mode='w') as writer:
            writer.write_all(results)
    
    autorace_score(output_log_path)
    return results

def autorace_evaluation(
    dataset: str = "gsm8k", 
    reasoning_model: str = "eval_model",
    output_log_dir: str = 'logs/auto_race',
    use_local_model: bool = False,
    local_model_path: Optional[str] = None,
    batch_size: int = BATCH_SIZE
):
    """Wrapper function to call either async or sync version"""
    if ASYNC_OPENAI_AVAILABLE:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(
            autorace_evaluation_async(dataset, reasoning_model, output_log_dir, use_local_model, local_model_path, batch_size)
        )
    else:
        # Fallback to synchronous version if async not available
        return autorace_evaluation_sync(dataset, reasoning_model, output_log_dir, use_local_model, local_model_path, batch_size)

def autorace_evaluation_sync(
    dataset: str = "gsm8k", 
    reasoning_model: str = "eval_model",
    output_log_dir: str = 'logs/auto_race',
    use_local_model: bool = False,
    local_model_path: Optional[str] = None,
    batch_size: int = BATCH_SIZE
):
    """Synchronous version of autorace evaluation"""
    predefined_datasets = ['gsm8k', 'strategyqa', 'aqua', 'cosmos', 'multistep_arithmetic', 'word_sorting', 'logical_deduction']
    
    if dataset not in predefined_datasets:
        print(f"Warning: The dataset '{dataset}' is not a predefined dataset.")
    if dataset not in PROMPT_TYPE_DICT:
        raise ValueError(f"dataset '{dataset}' is not in PROMPT_TYPE_DICT! Please add the prompt type to PROMPT_TYPE_DICT.")
    
    data_path = f'./data/{reasoning_model}/{dataset}.jsonl'
    assert os.path.exists(data_path), f'the output from {reasoning_model}: {data_path} does not exist!'
    
    output_log_dir = os.path.join(output_log_dir, reasoning_model, dataset)
    os.makedirs(output_log_dir, exist_ok=True)
    output_log_path = f'{output_log_dir}/autorace_eval.jsonl'
    
    print(f"Evaluating reasoning model: {reasoning_model} on dataset: {dataset}, output log path: {output_log_path}")
    
    # Load data
    data = pd.read_json(data_path, lines=True)
    
    # Load prompts
    with open('prompt.json') as f:
        prompts = json.load(f)
    
    # Process in batches with ThreadPoolExecutor for parallel processing
    results = []
    
    def process_example(index):
        reasoning_chain = data.loc[index, 'reasoning_chain']
        if not reasoning_chain.startswith('\n'):
            reasoning_chain = '\n' + reasoning_chain
        reasoning_chain = reasoning_chain.rstrip('\n\n.')
        
        raw_question = data.loc[index, 'question']
        raw_question = raw_question.replace('Q:', '')
        raw_question = raw_question.lstrip(' ')
        
        prompt = prompts[PROMPT_TYPE_DICT[dataset]].format(raw_question, reasoning_chain)
        prompt = prompt.replace('..', '.')
        
        cache_key = get_cache_key(prompt)
        cached_result = check_cache(cache_key)
        
        if cached_result:
            evaluation_result = cached_result
        else:
            try:
                if use_local_model and VLLM_AVAILABLE:
                    model = get_vllm_model(local_model_path)
                    output = model.generate(
                        [prompt],
                        SamplingParams(
                            temperature=TEMPERATURE,
                            top_p=TOP_P,
                            max_tokens=MAX_TOKENS,
                            n=NUM_RETURN_SEQUENCES,
                        )
                    )
                    evaluation_result = [output[0].outputs[0].text]
                else:
                    client = get_client()
                    messages = [{'role': 'user', 'content': prompt}]
                    response = client.chat.completions.create(
                        model=OPENAI_MODEL,
                        messages=messages,
                        max_tokens=MAX_TOKENS,
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        n=NUM_RETURN_SEQUENCES,
                    )
                    evaluation_result = [choice.message.content for choice in response.choices]
                
                save_to_cache(cache_key, evaluation_result)
            except Exception as e:
                print(f'An Error Occurred for index {index}: {e}, returning empty result')
                evaluation_result = ["ERROR"]
        
        return {
            'index': index,
            'evaluation_result': evaluation_result,
            'question': raw_question,
            'reasoning_chain': reasoning_chain,
            'answer': data.loc[index, 'answer'],
            'prompt': prompt
        }
    
    # Process examples in batches with parallel execution
    indices = list(range(len(data)))
    with ThreadPoolExecutor(max_workers=min(os.cpu_count(), 10)) as executor:
        for i in tqdm(range(0, len(indices), batch_size), desc="Processing batches"):
            batch_indices = indices[i:i+batch_size]
            batch_results = list(executor.map(process_example, batch_indices))
            results.extend(batch_results)
            
            # Save after each batch
            with jsonlines.open(output_log_path, mode='w') as writer:
                writer.write_all(results)
    
    autorace_score(output_log_path)
    return results

def test_evaluation_accuracy(output_name: str = time.strftime('%Y-%m-%d-%H-%M-%S')):
    """Test the accuracy of evaluation metrics using human labels as ground truth."""
    print("Start testing evaluation accuracy...")
    
    datasets = ['gsm8k','strategyqa','cosmos', 'multistep_arithmetic','word_sorting','logical_deduction']
    model = "eval_model"
    eval_dir = "./logs/auto_race"
    human_label_dir = "./data/eval_model"
    
    for dataset in datasets:
        if os.path.exists(f'{eval_dir}/{model}/{dataset}'):
            print(f'{eval_dir}/{model}/{dataset} exists, pass.')
        else:
            print(f'{eval_dir}/{model}/{dataset} does not exist, start autorace evaluation...')
            autorace_evaluation(dataset, model, eval_dir)
    
        human_label_path = os.path.join(human_label_dir, f'{dataset}.jsonl')
        evaluator_label_path = os.path.join(eval_dir, f'{model}/{dataset}/autorace_eval.jsonl')
        
        with jsonlines.open(human_label_path, mode='r') as reader:
            human_labels = list(reader)
        
        with jsonlines.open(evaluator_label_path, mode='r') as reader:
            evaluator_labels = list(reader)
            
        assert len(human_labels) >= len(evaluator_labels), f'there are unlabelled samples in {human_label_path} compared to {evaluator_label_path}!'

        total = len(evaluator_labels)
        score = 0
        correct_align_list = []
        incorrect_align_list = []
        incorrect_disagreement = []
        correct_disagreement = []
        
        for i in range(len(evaluator_labels)):
            output = evaluator_labels[i]['evaluation_result'][0]
            if 'INCORRECT' in output:
                if int(human_labels[i]['human_label']) == 0:
                    incorrect_align_list.append(i)
                    score += 1
                else:
                    incorrect_disagreement.append({
                        'index': i, 
                        'prompt': evaluator_labels[i]['prompt'], 
                        'answer': str(human_labels[i]['answer']), 
                        'human_label': str(human_labels[i]['human_label']), 
                        'evaluation_result': evaluator_labels[i]['evaluation_result']
                    })
            else:
                if int(human_labels[i]['human_label']) == 1:
                    correct_align_list.append(i)
                    score += 1
                else:
                    correct_disagreement.append({
                        'index': i, 
                        'prompt': evaluator_labels[i]['prompt'], 
                        'answer': str(human_labels[i]['answer']), 
                        'human_label': str(human_labels[i]['human_label']), 
                        'evaluation_result': evaluator_labels[i]['evaluation_result']
                    })

        output_dir = f'logs/error_analysis/{output_name}/{dataset}'
        correct_path = os.path.join(output_dir, 'correct_disagree')
        incorrect_path = os.path.join(output_dir, 'incorrect_disagree')
        align_score_log = os.path.join(output_dir, 'align_score.txt')
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(correct_path, exist_ok=True)
        os.makedirs(incorrect_path, exist_ok=True)

        for sample in incorrect_disagreement:
            with open(os.path.join(incorrect_path, f"{sample['index']}.txt"), 'w') as f:
                f.write('====================================\n')
                f.write(f'Index: {sample["index"]}\n')
                f.write(f'Answer: {sample["answer"]}\n')
                f.write(f'Human label: {sample["human_label"]}\n')
                f.write('====================================\n')
                f.write(f'Prompt: {sample["prompt"]}\n')
                f.write('====================================\n')
                f.write(f'Evaluation: {sample["evaluation_result"]}\n')
                
        for sample in correct_disagreement:
            with open(os.path.join(correct_path, f"{sample['index']}.txt"), 'w') as f:
                f.write('====================================\n')
                f.write(f'Index: {sample["index"]}\n')
                f.write(f'Answer: {sample["answer"]}\n')
                f.write(f'Human label: {sample["human_label"]}\n')
                f.write('====================================\n')
                f.write(f'Prompt: {sample["prompt"]}\n')
                f.write('====================================\n')
                f.write(f'Evaluation: {sample["evaluation_result"]}\n')

        align_score = score / total
        print(f'align score for {dataset}: {align_score:.2f}')
        with open(align_score_log, 'w') as f:
            f.write(f'Align score: {align_score:.2f}\n')
            f.write(f'Total: {total}\n')
            f.write(f'Correct: {score}\n')
            f.write(f'Incorrect: {total - score}\n')
            f.write(f'Correct align list: {correct_align_list}\n')
            f.write(f'Incorrect align list: {incorrect_align_list}\n')
            f.write(f'Correct disagreement list: {correct_disagreement}\n')
            f.write(f'Incorrect disagreement list: {incorrect_disagreement}\n')    

def main():
    parser = argparse.ArgumentParser(description='AutoRace Evaluation Tool')
    parser.add_argument('--gen_criteria', action='store_true', help='Generate criteria')
    parser.add_argument('--dataset', type=str, default='gsm8k', help='Dataset name')
    parser.add_argument('--example_wrong_chains', type=str, default='EXAMPLE_WRONG_CHAINS_AQUA.txt', help='Example wrong chains file')
    parser.add_argument('--reproduce_tab1', action='store_true', help='Reproduce Table 1 results')
    parser.add_argument('--reasoning_model', type=str, default='eval_model', help='Reasoning model directory')
    parser.add_argument('--output_log', type=str, default='logs/auto_race', help='Output log directory')
    parser.add_argument('--use_local_model', action='store_true', help='Use local model instead of OpenAI API')
    parser.add_argument('--local_model_path', type=str, help='Path to local model for vLLM')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size for processing examples')
    parser.add_argument('--openai_model', type=str, default=OPENAI_MODEL, help='OpenAI model to use')
    parser.add_argument('--max_tokens', type=int, default=MAX_TOKENS, help='Max tokens for generation')
    parser.add_argument('--max_concurrent', type=int, default=MAX_CONCURRENT_REQUESTS, help='Max concurrent API requests')
    
    args = parser.parse_args()
    
    # Update global settings if provided
    global OPENAI_MODEL, MAX_TOKENS, BATCH_SIZE, MAX_CONCURRENT_REQUESTS
    OPENAI_MODEL = args.openai_model
    MAX_TOKENS = args.max_tokens
    BATCH_SIZE = args.batch_size
    MAX_CONCURRENT_REQUESTS = args.max_concurrent
    
    if args.reproduce_tab1:
        test_evaluation_accuracy()
    elif args.gen_criteria:
        autorace_criterion(args.dataset, args.example_wrong_chains)
    else:
        autorace_evaluation(
            args.dataset, 
            args.reasoning_model, 
            args.output_log,
            args.use_local_model,
            args.local_model_path,
            args.batch_size
        )

if __name__ == '__main__':
    main()