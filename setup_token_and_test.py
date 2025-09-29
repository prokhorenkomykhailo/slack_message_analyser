#!/usr/bin/env python3
"""
Simple setup script to configure token and test Cohere access
"""

import os
import sys

def setup_and_test():
    """Setup token and test Cohere access"""
    
    print("🔧 Cohere Command R+ Setup and Test")
    print("=" * 40)
    
    
    if len(sys.argv) > 1:
        token = sys.argv[1]
        print(f"✅ Token provided (length: {len(token)})")
    else:
        print("❌ No token provided")
        print("\n📋 Usage:")
        print("   python setup_token_and_test.py YOUR_HUGGINGFACE_TOKEN")
        print("\n🔗 Get your token from: https://huggingface.co/settings/tokens")
        return False
    
    
    os.environ['HUGGINGFACE_TOKEN'] = token
    print("✅ Token set in environment")
    
    
    print("\n🔐 Testing authentication...")
    
    try:
        from huggingface_hub import login, whoami
        
        
        login(token=token)
        print("✅ Login successful")
        
        
        user_info = whoami()
        print(f"✅ Authenticated as: {user_info['name']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return False

def test_cohere_access():
    """Test access to Cohere models"""
    
    print("\n🧪 Testing Cohere model access...")
    
    models_to_test = [
        "CohereLabs/c4ai-command-r-plus-08-2024",
        "CohereLabs/c4ai-command-r-plus"
    ]
    
    for model_name in models_to_test:
        print(f"\n🔍 Testing {model_name}...")
        
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            
            
            files = api.list_repo_files(model_name)
            print(f"✅ Have access to {model_name}")
            print(f"   Available files: {len(files)}")
            return model_name
            
        except Exception as e:
            print(f"❌ No access to {model_name}: {e}")
            continue
    
    return None

def test_model_loading(model_name):
    """Test loading the model"""
    
    print(f"\n🚀 Testing model loading: {model_name}")
    
    try:
        from transformers import AutoTokenizer
        
        print("   Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("   ✅ Tokenizer loaded")
        
        
        test_text = "Hello, how are you?"
        tokens = tokenizer.encode(test_text)
        print(f"   ✅ Tokenization works ({len(tokens)} tokens)")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Model loading failed: {e}")
        return False

def create_simple_test():
    """Create a simple test script"""
    
    test_code = '''#!/usr/bin/env python3
"""
Simple Cohere test after authentication
"""

import os
import torch
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_cohere_clustering():
    """Test Cohere for Step 1 clustering"""
    
    print("🚀 Testing Cohere Step 1 Clustering")
    print("=" * 40)
    
    
    messages_file = "data/Synthetic_Slack_Messages.csv"
    if not os.path.exists(messages_file):
        print(f"❌ File not found: {messages_file}")
        return
    
    try:
        df = pd.read_csv(messages_file)
        messages = []
        
        for _, row in df.iterrows():
            messages.append({
                "id": row.get("id", len(messages) + 1),
                "user": row.get("user", "Unknown"),
                "content": row.get("content", ""),
                "channel": row.get("channel", "#general")
            })
        
        print(f"✅ Loaded {len(messages)} messages")
        
        
        test_messages = messages[:20]
        
        
        model_name = "CohereLabs/c4ai-command-r-plus-08-2024"
        print(f"🔄 Loading {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print("✅ Model loaded")
        
        
        messages_text = "\\n".join([
            f"{i+1}. {msg['user']}: {msg['content'][:100]}"
            for i, msg in enumerate(test_messages)
        ])
        
        prompt = f"""Group these messages into topic clusters:

{messages_text}

Return JSON with clusters containing message numbers, titles, and participants."""
        
        
        messages_chat = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages_chat, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        )
        
        print("🔄 Generating clusters...")
        
        with torch.no_grad():
            gen_tokens = model.generate(
                input_ids,
                max_new_tokens=800,
                do_sample=True,
                temperature=0.1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
        
        print("✅ Generation completed")
        print("\\n📋 Response:")
        print(response[:500] + "..." if len(response) > 500 else response)
        
        print("\\n🎉 Cohere Step 1 clustering test successful!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_cohere_clustering()
'''
    
    with open("simple_cohere_test.py", "w") as f:
        f.write(test_code)
    
    print("✅ Created simple_cohere_test.py")

def main():
    """Main function"""
    
    
    if not setup_and_test():
        return
    
    
    accessible_model = test_cohere_access()
    if not accessible_model:
        print("\n❌ No access to any Cohere model")
        print("Please request access at: https://huggingface.co/CohereLabs/c4ai-command-r-plus")
        return
    
    
    if not test_model_loading(accessible_model):
        print("\n❌ Model loading failed")
        return
    
    
    create_simple_test()
    
    print("\n🎉 Setup completed successfully!")
    print(f"\n✅ You have access to: {accessible_model}")
    print("\n📋 Next step:")
    print("   python simple_cohere_test.py")
    print("\nThis will test Cohere Step 1 clustering with your 200 messages")

if __name__ == "__main__":
    main()
