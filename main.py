import os
import subprocess
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch

# Set environment variable for better memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Initialize FastAPI app
app = FastAPI()

class CodeFile(BaseModel):
    path: str
    content: str

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Model is running on: {'GPU' if device == 'cuda' else 'CPU'}")

# Ensure git-lfs is installed
try:
    subprocess.run(["git", "lfs", "install"], check=True)
    print("✅ git-lfs is installed.")
except Exception as e:
    print(f"⚠️ Warning: git-lfs installation check failed. {str(e)}")

# Define model name
model_name = "microsoft/phi-2"

# Quantization configuration to reduce GPU memory usage
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True  # Use 8-bit quantization
)

try:
    print("📥 Downloading/loading model from Hugging Face Hub...")
    
    # Load tokenizer and model with quantization and GPU support
    tokenizer = AutoTokenizer.from_pretrained(model_name, force_download=True, token=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto",  # Auto-assign model layers between GPU & CPU
        trust_remote_code=True,
        quantization_config=quantization_config,
        offload_folder="offload"  # Directory for CPU offloading
    )
    
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"⚠️ Error loading model: {str(e)}")
    raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

# Initialize text generation pipeline
phi2_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def generate_documentation_for_code(code: str) -> str:
    """Generates Markdown documentation for the provided code."""
    prompt = (
" You are a technical documentation expert. Your task is to analyze the given code and generate well-structured Markdown documentation. Follow these instructions:"
"1. Overview: Summarize the purpose of the file concisely."
"2. Functions & Methods: List all functions/methods along with a short explanation of their purpose."
"3. Classes & Components: List any classes, their attributes, and their key methods."
"4. Key Variables: Identify important variables/constants used in the file."
"5. Comments & Annotations: Extract any relevant comments that clarify functionality."
"7. If any of the above is not applicable for the given code simply skip that section"
"Ensure the documentation follows proper Markdown syntax, with clear headers, bullet points, and code blocks where necessary."
"Avoid unnecessary examples or exercises unless they exist in the original code. Do not fabricate details or assume functionality beyond what is provided."
"Simply document the given code so that it is easy for a person to see and understand what is going on in the code"
"Now analyze the following code and generate Markdown documentation:"
"Code:"
f"{code}"
    )
    result = phi2_generator(prompt, max_new_tokens=300, do_sample=True, temperature=0.8)
    generated_text = result[0]["generated_text"]
    documentation = generated_text.replace(prompt, "").strip()
    return documentation

@app.post("/document")
def document_codebase(code_files: list[CodeFile]):  
    """Processes multiple code files and generates documentation."""
    print(code_files)
    docs = {}
    for code_file in code_files:
        try:
            print(f"📄 Documenting {code_file.path} ...")
            documentation = generate_documentation_for_code(code_file.content)
            docs[code_file.path] = documentation
        except Exception as e:
            print(f"⚠️ Error processing {code_file.path}: {e}")
            docs[code_file.path] = f"Error generating documentation: {str(e)}"

    # Format the Markdown documentation
    md_content = "# Codebase Documentation\n\n"
    for file_path, doc in docs.items():
        md_content += f"## {file_path}\n\n{doc}\n\n"
    return {"documentation": md_content}
