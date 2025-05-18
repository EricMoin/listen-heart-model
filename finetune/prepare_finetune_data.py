import json
import os
import random
from tqdm import tqdm

def load_qa_pairs(input_file):
    """Load QA pairs from JSON file"""
    print(f"Loading QA pairs from {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} QA pairs")
    return data

def convert_to_qwen_format(qa_pairs, output_file, train_ratio=0.9):
    """Convert QA pairs to Qwen fine-tuning format and split into train/val"""
    # Shuffle the data for random split
    random.seed(42)  # For reproducibility
    random.shuffle(qa_pairs)
    
    # Split into train and validation sets
    split_idx = int(len(qa_pairs) * train_ratio)
    train_data = qa_pairs[:split_idx]
    val_data = qa_pairs[split_idx:]
    
    print(f"Split data into {len(train_data)} training samples and {len(val_data)} validation samples")
    
    # Create the formatted data
    train_formatted = []
    val_formatted = []
    
    # Function to format a single QA pair
    def format_qa_pair(qa_pair):
        question = qa_pair['question']
        answer = qa_pair['answer']
        emotion = qa_pair.get('emotion', 'neutral')  # Default to neutral if no emotion
        
        # Create system message based on emotion
        emotion_prompts = {
            'neutral': "你是一个温和的倾听者。对方现在情绪平静。",
            'calm': "你是一个平和的陪伴者。对方现在心情平静。",
            'happy': "你是一个温暖的分享者。对方现在心情愉快。",
            'sad': "你是一个温柔的安慰者。对方现在感到悲伤。",
            'angry': "你是一个冷静的疏导者。对方现在感到愤怒。",
            'fearful': "你是一个安心的守护者。对方现在感到害怕。",
            'disgust': "你是一个理解的倾听者。对方现在感到厌恶。",
            'surprised': "你是一个好奇的分享者。对方现在感到惊讶。"
        }
        
        system_message = emotion_prompts.get(emotion, emotion_prompts['neutral'])
        
        # Format for Qwen training
        formatted = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
        }
        
        return formatted
    
    print("Formatting training data...")
    for qa_pair in tqdm(train_data):
        train_formatted.append(format_qa_pair(qa_pair))
    
    print("Formatting validation data...")
    for qa_pair in tqdm(val_data):
        val_formatted.append(format_qa_pair(qa_pair))
    
    # Save the formatted data
    train_output = os.path.join(os.path.dirname(output_file), "train_" + os.path.basename(output_file))
    val_output = os.path.join(os.path.dirname(output_file), "val_" + os.path.basename(output_file))
    
    with open(train_output, 'w', encoding='utf-8') as f:
        json.dump(train_formatted, f, ensure_ascii=False, indent=2)
    
    with open(val_output, 'w', encoding='utf-8') as f:
        json.dump(val_formatted, f, ensure_ascii=False, indent=2)
    
    print(f"Saved formatted training data to {train_output}")
    print(f"Saved formatted validation data to {val_output}")
    
    # Also save as JSONL format (one JSON object per line) for easier loading
    train_jsonl = train_output.replace('.json', '.jsonl')
    val_jsonl = val_output.replace('.json', '.jsonl')
    
    print("Converting to JSONL format...")
    with open(train_jsonl, 'w', encoding='utf-8') as f:
        for item in train_formatted:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    with open(val_jsonl, 'w', encoding='utf-8') as f:
        for item in val_formatted:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Saved JSONL training data to {train_jsonl}")
    print(f"Saved JSONL validation data to {val_jsonl}")
    
    return {
        'train_json': train_output,
        'val_json': val_output,
        'train_jsonl': train_jsonl,
        'val_jsonl': val_jsonl,
        'train_samples': len(train_data),
        'val_samples': len(val_data)
    }

if __name__ == "__main__":
    # Input file - emotion_labeled_combined_qa.json
    input_file = "../output/fixed_qa_pairs/emotion_labeled_combined_qa.json"
    
    # Output file for formatted data
    output_file = "formatted_qa_data.json"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist.")
        print("Trying alternative path...")
        input_file = "../output/qa_pairs/emotion_labeled_combined_qa.json"
        if not os.path.exists(input_file):
            print(f"Error: Alternative input file {input_file} does not exist.")
            exit(1)
    
    # Load QA pairs
    qa_pairs = load_qa_pairs(input_file)
    
    # Convert to Qwen format
    result = convert_to_qwen_format(qa_pairs, output_file)
    
    print("\nData preparation completed!")
    print(f"Total QA pairs: {len(qa_pairs)}")
    print(f"Training samples: {result['train_samples']}")
    print(f"Validation samples: {result['val_samples']}")
    print(f"Files generated:")
    print(f"  - {result['train_json']}")
    print(f"  - {result['val_json']}")
    print(f"  - {result['train_jsonl']}")
    print(f"  - {result['val_jsonl']}") 