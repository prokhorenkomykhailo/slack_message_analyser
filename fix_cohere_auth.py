#!/usr/bin/env python3
"""
Fix authentication issues with Cohere Command R+ model
This script helps resolve token and authentication problems
"""

import os
import sys
import subprocess
from huggingface_hub import HfApi, login, whoami

def check_current_auth():
    """Check current authentication status"""
    print("🔐 Checking current Hugging Face authentication...")
    
    try:
        user_info = whoami()
        print(f"✅ Currently authenticated as: {user_info['name']}")
        print(f"   User ID: {user_info['id']}")
        return True, user_info
    except Exception as e:
        print(f"❌ Not authenticated: {e}")
        return False, None

def check_model_access(model_name="CohereLabs/c4ai-command-r-plus"):
    """Check if we have access to the specific model"""
    print(f"🔍 Checking access to {model_name}...")
    
    try:
        api = HfApi()
        model_info = api.model_info(model_name)
        
        if model_info.gated:
            print(f"✅ Model {model_name} is gated - access control active")
            
            # Check if we have access
            try:
                # Try to access model files
                api.list_repo_files(model_name)
                print(f"✅ You have access to {model_name}")
                return True
            except Exception as e:
                if "401" in str(e) or "403" in str(e):
                    print(f"❌ Access denied to {model_name}")
                    print(f"   Error: {e}")
                    return False
                else:
                    print(f"⚠️  Unexpected error: {e}")
                    return False
        else:
            print(f"✅ Model {model_name} is public - no access control")
            return True
            
    except Exception as e:
        print(f"❌ Error checking model access: {e}")
        return False

def reauthenticate():
    """Re-authenticate with Hugging Face"""
    print("🔄 Re-authenticating with Hugging Face...")
    
    try:
        # Use the login function
        token = input("Enter your Hugging Face token: ").strip()
        
        if not token:
            print("❌ No token provided")
            return False
        
        # Login with the token
        login(token=token)
        
        # Verify authentication
        user_info = whoami()
        print(f"✅ Successfully authenticated as: {user_info['name']}")
        return True
        
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return False

def test_model_loading(model_name="CohereLabs/c4ai-command-r-plus"):
    """Test actually loading the model"""
    print(f"🧪 Testing model loading: {model_name}")
    
    try:
        from transformers import AutoTokenizer
        
        print("   Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("   ✅ Tokenizer loaded successfully")
        
        # Test tokenizer functionality
        test_text = "Hello, how are you?"
        tokens = tokenizer.encode(test_text)
        print(f"   ✅ Tokenization test passed ({len(tokens)} tokens)")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Model loading failed: {e}")
        return False

def fix_environment_variables():
    """Fix environment variables"""
    print("🔧 Checking environment variables...")
    
    token = os.getenv('HUGGINGFACE_TOKEN')
    if token:
        print(f"✅ HUGGINGFACE_TOKEN is set (length: {len(token)})")
        
        # Test if token works
        try:
            api = HfApi(token=token)
            user_info = api.whoami()
            print(f"✅ Token is valid for user: {user_info['name']}")
            return True
        except Exception as e:
            print(f"❌ Token is invalid: {e}")
            return False
    else:
        print("❌ HUGGINGFACE_TOKEN not set")
        return False

def create_working_cohere_test():
    """Create a working test for Cohere model"""
    
    test_code = '''#!/usr/bin/env python3
"""
Working test for Cohere Command R+ after authentication fix
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_cohere_model():
    """Test Cohere Command R+ model loading and basic functionality"""
    
    model_name = "CohereLabs/c4ai-command-r-plus"
    
    print(f"🧪 Testing {model_name}")
    print("=" * 50)
    
    try:
        # Step 1: Load tokenizer
        print("1️⃣ Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("   ✅ Tokenizer loaded")
        
        # Step 2: Test tokenizer
        print("2️⃣ Testing tokenizer...")
        test_messages = [{"role": "user", "content": "Hello, how are you?"}]
        input_ids = tokenizer.apply_chat_template(
            test_messages, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        )
        print("   ✅ Chat template applied")
        
        # Step 3: Load model (small test first)
        print("3️⃣ Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print("   ✅ Model loaded")
        
        # Step 4: Test generation
        print("4️⃣ Testing generation...")
        with torch.no_grad():
            gen_tokens = model.generate(
                input_ids,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.3,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
        print("   ✅ Generation successful")
        print(f"   Response: {response[:100]}...")
        
        print("\\n🎉 Cohere Command R+ is working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        
        if "401" in str(e) or "403" in str(e):
            print("\\n🔧 Authentication issue detected:")
            print("   1. Make sure you're logged in with the correct account")
            print("   2. Verify you have access to the gated model")
            print("   3. Try: huggingface-cli login")
        elif "gated" in str(e).lower():
            print("\\n🔧 Gated model issue:")
            print("   1. Request access at: https://huggingface.co/CohereLabs/c4ai-command-r-plus")
            print("   2. Wait for approval")
        
        return False

if __name__ == "__main__":
    test_cohere_model()
'''
    
    with open("test_cohere_working.py", "w") as f:
        f.write(test_code)
    
    print("✅ Created test_cohere_working.py")

def main():
    """Main function to fix authentication issues"""
    
    print("🔧 Cohere Command R+ Authentication Fix")
    print("=" * 50)
    
    # Step 1: Check current authentication
    is_auth, user_info = check_current_auth()
    
    # Step 2: Check environment variables
    env_ok = fix_environment_variables()
    
    # Step 3: Check model access
    if is_auth:
        access_ok = check_model_access()
    else:
        access_ok = False
    
    # Step 4: Test model loading
    if access_ok:
        loading_ok = test_model_loading()
    else:
        loading_ok = False
    
    # Summary and next steps
    print("\\n📋 DIAGNOSIS SUMMARY:")
    print("=" * 30)
    
    if loading_ok:
        print("🎉 Everything is working! You can proceed with Cohere evaluation.")
        return
    
    print("🔍 Issues found:")
    
    if not is_auth:
        print("   ❌ Not authenticated with Hugging Face")
        print("   💡 Solution: Run 'huggingface-cli login' or re-authenticate")
    
    if not env_ok:
        print("   ❌ HUGGINGFACE_TOKEN environment variable issue")
        print("   💡 Solution: Set token with 'export HUGGINGFACE_TOKEN=your_token'")
    
    if not access_ok:
        print("   ❌ No access to Cohere Command R+ model")
        print("   💡 Solution: Request access at the Hugging Face model page")
    
    # Create working test
    create_working_cohere_test()
    
    print("\\n🔧 FIXES TO TRY:")
    print("1. Re-authenticate:")
    print("   huggingface-cli login")
    print("   # OR")
    print("   export HUGGINGFACE_TOKEN=your_token_here")
    
    print("\\n2. Test with the working script:")
    print("   python test_cohere_working.py")
    
    print("\\n3. If still having issues, try the latest model:")
    print("   # Use: CohereLabs/c4ai-command-r-plus-08-2024")

if __name__ == "__main__":
    main()
