from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import subprocess

# Initialize FastAPI app
app = FastAPI()

class CodeFile(BaseModel):
    path: str
    content: str

# Force the model to run on CPU
device = "cpu"
print(f"🚀 Model is running on: {'GPU' if device == 'cuda' else 'CPU'}")

# Ensure git-lfs is installed
try:
    subprocess.run(["git", "lfs", "install"], check=True)
    print("✅ git-lfs is installed.")
except Exception as e:
    print(f"⚠️ Warning: git-lfs installation check failed. {str(e)}")

# Define model name
model_name = "microsoft/phi-2"

try:
    print("📥 Downloading/loading model from Hugging Face Hub...")
    
    # Load tokenizer and model from Hugging Face Hub
    tokenizer = AutoTokenizer.from_pretrained(model_name, force_download=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map={"": "cpu"},
        trust_remote_code=True
    )
    
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"⚠️ Error loading model: {str(e)}")
    raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

# Initialize text generation pipeline
phi2_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def generate_documentation_for_code(code: str) -> str:
    prompt = (
        "You are a technical documentation expert skilled in multiple programming languages "
        "and frameworks. Your task is to generate detailed and visually appealing Markdown documentation "
        "for the given code. Follow these instructions:\n\n"
        "1. **File Header:**\n"
        "   - Start with a top-level header (using '#') that displays the file name (extracted from the code context).\n"
        "   - On the next line, display the full file path in plain text.\n\n"
        "2. **Documentation Sections:**\n"
        "   - **Overview:** Provide a brief summary of the purpose of the code.\n"
        "   - **Functions:** If there are any functions, list them with a short description of their behavior and parameters.\n"
        "   - **Classes:** If there are classes, list them with a description of their purpose and key methods.\n"
        "   - **Key Variables & Components:** List any important variables, constants, or modules used.\n"
        "   - **Comments/Annotations:** Include any relevant comments that explain the logic.\n\n"
        "3. **Formatting:**\n"
        "   - Use proper Markdown formatting (headers, bullet points, code blocks) so that the documentation is "
        "      clear, easy to read, and visually appealing.\n"
        "   - Do not add any fabricated sections or details that are not present in the code.\n\n"
        "4. **Generic and Accurate:**\n"
        "   - The output should be generic enough to apply to code written in any programming language or framework.\n"
        "   - Only document the code that is provided without inventing routes, endpoints, or features that do not exist.\n\n"
        "Below is the code. Generate the documentation in Markdown format as described above.\n\n"
        "Code:\n"
        f"{code}\n\n"
        "Documentation:"
    )
    result = phi2_generator(prompt, max_new_tokens=300, do_sample=True, temperature=0.7)
    generated_text = result[0]["generated_text"]
    documentation = generated_text.replace(prompt, "").strip()
    return documentation

@app.post("/document")
def document_codebase(code_files: list[CodeFile]):  # Now accepting a list directly
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

    md_content = "# Codebase Documentation\n\n"
    for file_path, doc in docs.items():
        md_content += f"## {file_path}\n\n{doc}\n\n"
    return {"documentation": md_content}
